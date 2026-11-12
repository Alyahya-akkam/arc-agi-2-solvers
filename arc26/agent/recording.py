"""An engine client that logs every call, and the per-reply health judgment (P6.5b).

Why: run ``P6.3-05`` read its engine's health (errors, degenerate and empty replies, where the
reasoning arrived, the finish reasons) and its decode rate from one row per call in
``engine_calls.jsonl``; its requests asked vLLM for each reply's token ids and counted each opening
prompt with ``/tokenize``. The hybrid's shot (:mod:`arc26.agent.shot`) sends the same requests,
so its client is the spike's, moved here.

- :class:`RecordingEngine`: P6.2's :class:`~arc26.agent.engine.OpenAIEngine` that appends one row
  per chat call to ``engine_calls.jsonl`` and one per prompt count to ``prompt_counts.jsonl``, and
  keeps each attempt's rows for the caller.
- :func:`recording_engine`: builds it with the spike's request fields (``return_token_ids``, the
  ``/tokenize`` URL).
- :func:`call_health`, :func:`token_stats`, :func:`reasoning_location`: one reply's judgment.
"""

# Moved from scripts/agent_spike.py (task P6.3, commit acac69d, the code run P6.3-05 ran):
# JsonlWriter, token_stats, call_health, and reasoning_location without change (a test checks
# that each source is identical to the script's); RecordingEngine and recording_engine are the
# class and the config that make_engine built inside the script, with the forcing check's prompt
# capture left out and each attempt's rows kept.

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arc26.agent.engine import EngineConfig, OpenAIEngine
from arc26.agent.loop import current_attempt
from arc26.timing import TimingLogger


class JsonlWriter:
    """Append-only JSONL files shared by threads; each record is flushed as written."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append(self, path: Path, record: Mapping[str, Any]) -> None:
        """Append one record as one line.

        Args:
            path: The JSONL file.
            record: A JSON-serializable mapping.
        """
        line = json.dumps(record, default=str) + "\n"
        with self._lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def token_stats(
    token_ids: Sequence[int],
    special_min_id: int,
    max_run: int,
    text: str = "",
    eos_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Measure repetition in one output: the longest run of one token, special tokens excluded.

    Why: NaN logits make a sampler emit one token over and over (often id 0); a run this long is
    the cheap signal of a broken engine, the kill criterion of the brief. Only the tokens before
    the first end-of-sequence id count: with ``ignore_eos`` a model that has finished keeps
    generating, and what follows its end may legitimately repeat.

    Args:
        token_ids: The generated tokens.
        special_min_id: Ids at or above this are special and break runs.
        max_run: Run length that marks the output degenerate.
        text: The generated text, used for a character run when no ids came back.
        eos_ids: End-of-sequence ids; the count stops at the first one.

    Returns:
        ``max_run``, ``max_run_token``, ``token0_count``, ``degenerate``, ``source``, and
        ``eos_at`` (the index of the first end-of-sequence id, or None).
    """
    stops = set(eos_ids)
    eos_at = next((i for i, token in enumerate(token_ids) if token in stops), None)
    if eos_at is not None:
        token_ids = token_ids[:eos_at]
    best, best_token, run, previous = 0, None, 0, None
    for token in token_ids:
        if token >= special_min_id:
            run, previous = 0, None
            continue
        run = run + 1 if token == previous else 1
        previous = token
        if run > best:
            best, best_token = run, token
    source = "token_ids"
    if not token_ids and text:
        source = "characters"
        for match in re.finditer(r"(.)\1*", text, re.DOTALL):
            length = len(match.group(0))
            if length > best:
                best, best_token = length, match.group(1)
    return {
        "max_run": best,
        "max_run_token": best_token,
        "token0_count": sum(1 for t in token_ids if t == 0),
        "degenerate": best >= max_run,
        "source": source,
        "eos_at": eos_at,
    }


def call_health(
    token_ids: Sequence[int],
    text: str,
    thought: str,
    max_tokens: int | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Judge one engine reply for the health check: repetition, emptiness, NaN.

    A reply is degenerate when its longest run of one non-special token before the first
    end-of-sequence id reaches ``max_run`` (``token_stats``, as stage 1 counts it; a run of token
    0 is the usual sign of NaN logits). An empty reply (no answer and no reasoning) is counted
    apart as ``empty``, a model behavior reported, not a degenerate output: run ``P6.3-04``'s
    whole Fault came from one forced reply that stopped after one token, with no run and no
    token 0. A request capped under 16 tokens (the preserve check's one-token turns) may come
    back blank, so it is never counted empty.

    Args:
        token_ids: The reply's token ids (vLLM's ``return_token_ids``), empty when absent.
        text: The answer text.
        thought: The reasoning text.
        max_tokens: The request's cap on generated tokens.
        settings: The config's ``degenerate`` section.

    Returns:
        ``max_run``, ``token0_count``, ``empty``, ``degenerate``, and the run's ``source``.
    """
    stats = token_stats(
        token_ids, settings["special_min_id"], settings["max_run"], thought + text,
        settings["eos_ids"],
    )  # fmt: skip
    empty = (max_tokens or 0) >= 16 and not text.strip() and not thought.strip()
    return {
        "max_run": stats["max_run"],
        "token0_count": stats["token0_count"],
        "run_source": stats["source"],
        "empty": empty,
        "degenerate": bool(stats["degenerate"]),
    }


def reasoning_location(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return where one chat completion message carries its reasoning.

    Why: vLLM 0.31 names the field ``reasoning``, SGLang and older vLLM ``reasoning_content``,
    and a server without a reasoning parser leaves the thinking inline in ``content`` up to
    ``</think>``. P6.2's client reads all three; the spike records which one each request used.

    Args:
        message: ``choices[0].message`` of a non-streaming response.

    Returns:
        ``location``: the first of ``"reasoning"``, ``"reasoning_content"``, ``"inline"``,
        ``"none"`` that applies; ``fields``: the reasoning fields present and non-empty;
        ``think_close_in_content``: whether ``content`` holds ``</think>``.
    """
    fields = [name for name in ("reasoning", "reasoning_content") if message.get(name)]
    content = message.get("content")
    inline = isinstance(content, str) and "</think>" in content
    location = fields[0] if fields else ("inline" if inline else "none")
    return {"location": location, "fields": fields, "think_close_in_content": inline}


class RecordingEngine(OpenAIEngine):
    """``OpenAIEngine`` that logs each call: reasoning location, health, tokens, errors.

    Each chat call appends one row to ``calls_path``: the attempt (from the loop's context), where
    the reply's reasoning arrived (:func:`reasoning_location`), how many assistant turns the
    request sent back with reasoning and with which ``preserve_thinking``, the call's thinking
    switch (False on a forcing turn), :func:`call_health` of the reply, the token counts, the
    finish reason, and the seconds; a failed or cancelled call its error. Each opening prompt count
    appends one row to ``counts_path``. The rows of every call are also kept per attempt id in
    :attr:`calls`, so the caller can tell which attempts had a degenerate reply.

    Args:
        config: The client's settings.
        timing: Where each call's timing record goes, or None.
        calls_path: The JSONL of chat calls (``engine_calls.jsonl``).
        counts_path: The JSONL of prompt counts (``prompt_counts.jsonl``).
        degenerate: The config's ``degenerate`` section (``max_run``, ``special_min_id``,
            ``eos_ids``).
        writer: The thread-safe appender; a new one when None.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        timing: TimingLogger | None,
        calls_path: Path,
        counts_path: Path,
        degenerate: Mapping[str, Any],
        writer: JsonlWriter | None = None,
    ) -> None:
        super().__init__(config, timing=timing)
        self.calls_path = Path(calls_path)
        self.counts_path = Path(counts_path)
        self.degenerate = dict(degenerate)
        self.writer = writer if writer is not None else JsonlWriter()
        self.calls: dict[str | None, list[dict[str, Any]]] = defaultdict(list)

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        try:
            self.writer.append(path, row)
        except OSError:  # a log that cannot be written must not fail the call
            pass

    async def count_prompt_tokens(self, messages: Sequence[Any], **kwargs: Any) -> int:
        """Count the opening prompt and log it to ``prompt_counts.jsonl``.

        Args:
            messages: The conversation, as :meth:`complete` would send it.
            **kwargs: As :meth:`OpenAIEngine.count_prompt_tokens`.

        Returns:
            The server's count.

        Raises:
            EngineError: As :meth:`OpenAIEngine.count_prompt_tokens`.
        """
        context = current_attempt()
        row: dict[str, Any] = {"attempt_id": None if context is None else context.attempt_id}
        try:
            count = await super().count_prompt_tokens(messages, **kwargs)
        except BaseException as err:
            row["error"] = f"{type(err).__name__}: {err}"[:300]
            self._append(self.counts_path, row)
            raise
        row.update(count=count, error=None)
        self._append(self.counts_path, row)
        return count

    async def _post(self, body: Mapping[str, Any]) -> Any:
        began = time.monotonic()
        context = current_attempt()
        field = self.config.reasoning_field
        sent = [m for m in body["messages"] if m.get("role") == "assistant" and m.get(field)]
        template = body.get("chat_template_kwargs") or {}
        row: dict[str, Any] = {
            "puzzle_id": None if context is None else context.task_id,
            "attempt_id": None if context is None else context.attempt_id,
            "reasoning_sent": len(sent),
            "preserve_thinking": template.get("preserve_thinking"),
            # False on a forcing turn; absent means the default, on.
            "thinking": template.get(self.config.thinking_kwarg),
        }
        if "thinking_token_budget" in body:  # only a request with in-request forcing (P6.9)
            row["thinking_token_budget"] = body["thinking_token_budget"]
        try:
            data = await super()._post(body)
        except BaseException as err:
            row.update(error=f"{type(err).__name__}: {err}"[:500])
            row["seconds"] = round(time.monotonic() - began, 3)
            row["ended"] = round(time.time(), 3)
            self.calls[row["attempt_id"]].append(row)
            self._append(self.calls_path, row)
            raise
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        usage = (data.get("usage") if isinstance(data, dict) else None) or {}
        text = (message.get("content") or "") if isinstance(message.get("content"), str) else ""
        thought = message.get("reasoning") or message.get("reasoning_content") or ""
        stats = call_health(
            choice.get("token_ids") or [], text, thought, body.get("max_tokens"), self.degenerate
        )
        row.update(
            error=None,
            **reasoning_location(message),
            **stats,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            seconds=round(time.monotonic() - began, 3),
            ended=round(time.time(), 3),
        )
        self.calls[row["attempt_id"]].append(row)
        self._append(self.calls_path, row)
        return data


def recording_engine(
    root_url: str,
    model: str,
    max_connections: int,
    log_dir: Path,
    degenerate: Mapping[str, Any],
    *,
    return_token_ids: bool = True,
    count_prompts: bool = True,
    **client: Any,
) -> RecordingEngine:
    """Build the shot's engine client as run ``P6.3-05``'s ``make_engine`` built it.

    Args:
        root_url: The server's root, for example ``http://127.0.0.1:8000``.
        model: The served model name.
        max_connections: Connections in flight (the wave's tasks).
        log_dir: Where ``engine_calls.jsonl``, ``prompt_counts.jsonl``, and the timing log go.
        degenerate: The config's ``degenerate`` section.
        return_token_ids: Ask vLLM for each reply's token ids (a top-level request field).
        count_prompts: Count each opening prompt with vLLM's ``/tokenize``.
        **client: Further ``EngineConfig`` fields (timeouts, retries).

    Returns:
        The engine; nothing connects until the first call.
    """
    extra = {"return_token_ids": True} if return_token_ids else {}
    config = EngineConfig(
        base_url=root_url + "/v1",
        # The opening prompt's count: vLLM's /tokenize, outside /v1.
        tokenize_url=root_url + "/tokenize" if count_prompts else None,
        model=model,
        max_connections=max_connections,
        extra_body=extra,
        **client,
    )
    return RecordingEngine(
        config,
        timing=TimingLogger(log_dir, "engine"),
        calls_path=log_dir / "engine_calls.jsonl",
        counts_path=log_dir / "prompt_counts.jsonl",
        degenerate=degenerate,
    )
