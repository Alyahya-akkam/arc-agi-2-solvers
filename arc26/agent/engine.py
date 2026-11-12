r"""An async client for an OpenAI-compatible chat server: vLLM or SGLang on localhost (P6.2).

:class:`OpenAIEngine` implements :class:`~arc26.agent.types.Engine` over
``POST <base_url>/chat/completions`` (non-streaming) with one ``httpx.AsyncClient`` per event
loop (a client is bound to the loop that made it, so a second ``asyncio.run`` gets a new one);
it bounds the connections in flight and keeps none idle, so nothing lingers after a loop ends.
Connection failures are retried with exponential backoff; HTTP errors and timeouts are not (a
retried generation would run twice). There is no read timeout by default: a call waits for
the whole generation, and the loop's deadline bounds it. Token counts come from the response's
``usage``; the reasoning text from ``message.reasoning_content`` (SGLang, older vLLM) or
``message.reasoning`` (vLLM 0.31). A server started without a reasoning parser leaves the
reasoning in ``content``: when no reasoning field arrives and the content holds ``</think>``, the
text before the last ``</think>`` is taken as the reasoning, and only the rest as the answer.
Each call appends ``{puzzle_id, op: "engine", seconds, tokens}`` to a timing log
(:mod:`arc26.timing`; ``tokens`` are the generated tokens), with the task id taken from the
loop's attempt context.

Qwen3.8 (model card and chat template of ``Qwen/Qwen3.8-27B`` rev ``1d4bf0f2``, identical in
``RedHatAI/Qwen3.8-27B-INT4`` rev ``91bd022d``; vLLM 0.31.0 and SGLang 0.5.21 sources):

- Thinking switch: the template variable ``enable_thinking`` (on by default), sent in
  ``chat_template_kwargs``; ``false`` prefills an empty think block (the generation prompt ends
  ``<think>\n\n</think>\n\n``). ``EngineConfig.thinking`` sets it for every call, and
  ``complete(thinking=...)`` for one call (the loop's forcing turn).
- Effort: ``reasoning_effort``, one of ``xhigh`` (default), ``medium``, ``low``; the template
  raises on any other value while thinking is on. The card sends it as the top-level request
  field; vLLM 0.31 and SGLang 0.5.21 both forward that field into the template (and vLLM's
  ``"none"`` turns thinking off). The vLLM recipe sends it in ``chat_template_kwargs``; both
  transports are config values here. The template reads the effort only while thinking is on
  (its check and its system-prompt instruction sit inside ``enable_thinking``'s test), so it
  cannot raise on a call with thinking off, and such a call's system message would lose the
  effort instruction: its prompt would differ from the thinking turns' in its first tokens, and
  the engine would prefill the whole conversation again (up to 30k tokens per forcing turn). On a
  call whose own switch is off the client therefore sends no effort and writes the instruction
  at the head of the system message itself (``QWEN38_EFFORT_NOTES``, the template's text), which
  renders the same system block.
- ``preserve_thinking`` (on by default in the template) re-renders earlier assistant turns with
  their ``reasoning_content``. The engine sends an assistant message's ``Message.reasoning`` as
  ``reasoning_content`` and sets ``preserve_thinking`` to whether any earlier turn carries
  reasoning; the loop keeps reasoning only under ``Budget.preserve_thinking`` (off by default).
  Without reasoning, earlier turns render as content only, not with empty think blocks that
  mimic the non-thinking mode. The template's test for this does not read ``enable_thinking``,
  so earlier reasoning renders with thinking off too; it wraps each kept reasoning, trimmed, in
  ``<think>\n`` and ``\n</think>\n\n``, so a reasoning cut at the cap, which has no closing
  tag, is closed by the template.
- A reply generated with thinking off holds no reasoning; if the server's reasoning parser
  still files its text as reasoning (a parser that treats output without ``</think>`` as
  thinking), the client moves that text to the answer.
- Sampling for thinking mode: temperature 1.0, top_p 0.95, top_k 20 (``generation_config.json``,
  which vLLM applies by default).
- Thinking budget (task P6.9, option 2b (iii)): vLLM 0.31.0's chat request takes a top-level
  ``thinking_token_budget`` (``vllm/entrypoints/openai/chat_completion/protocol.py``, a
  ``SamplingParams`` field). With ``--reasoning-parser qwen3`` the server's ``ReasoningConfig``
  is filled from the parser (``<think>``, ``</think>``; ``vllm/config/reasoning.py``), and
  ``--reasoning-config`` may give a ``reasoning_end_str`` with a transition phrase before
  ``</think>``. Once the reasoning since the last ``<think>`` (the generation prompt's, counted
  from the prompt) reaches the budget, the sampler forces that string token by token and the
  model then writes its answer in the same request
  (``vllm/v1/worker/gpu/sample/thinking_budget.py``, Model Runner V2, which ``P6.3-05``'s boot
  used). ``complete(thinking_token_budget=...)`` sends it for one call; the loop passes it only
  when ``Budget.forcing_in_request`` is on, so an engine without the argument serves every
  other call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from arc26.agent.loop import current_attempt
from arc26.agent.types import Completion, Message
from arc26.timing import TimingLogger

logger = logging.getLogger(__name__)

ENGINE_OP = "engine"
# Failures where no response arrived because the connection did not open or was dropped before
# a status line (a server still booting, or restarting). Read timeouts are not among them.
RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# The system-prompt instruction Qwen3.8's chat template adds for each effort while thinking is on
# (rev 91bd022d; "medium" adds none). A call with thinking off gets it in its system message from
# the client instead, so its prompt starts with the same tokens as its thinking turns' and the
# engine's prefix cache serves it.
QWEN38_EFFORT_NOTES = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
        "and clarity in the final answer."
    ),
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
        "the conclusion without unnecessary elaboration."
    ),
}


class EngineError(RuntimeError):
    """An engine call failed: HTTP error, timeout, exhausted retries, or a malformed reply."""


@dataclass(frozen=True)
class EngineConfig:
    """Settings of an :class:`OpenAIEngine`.

    Attributes:
        base_url: The server's OpenAI API root, for example ``"http://127.0.0.1:8000/v1"``.
        model: The served model name (vLLM's ``--served-model-name``, else the model path).
        api_key: A bearer token if the server requires one (vLLM ``--api-key``), else None.
        max_connections: The most connections in flight; set it to at least the agent
            concurrency.
        connect_timeout_s: Seconds to open a connection (also the write timeout).
        read_timeout_s: Seconds to wait for a response, or None (the default) for no limit: a
            non-streaming call waits for the whole generation, and the loop's deadline bounds it.
        pool_timeout_s: Seconds to wait for a free pooled connection, or None (the default):
            with more calls in flight than ``max_connections``, calls queue for a connection.
        max_retries: Retries after a connection failure.
        backoff_s: The first retry's delay; it doubles per retry, with jitter.
        backoff_max_s: The cap on one retry's delay.
        thinking: The thinking switch, or None for the template's default (on).
        thinking_kwarg: The template variable of the switch (Qwen3.8: ``enable_thinking``).
        effort_kwarg: The name of the effort parameter (Qwen3.8: ``reasoning_effort``).
        effort_transport: ``"request"`` sends the effort as a top-level request field (the
            card's way); ``"template"`` puts it in ``chat_template_kwargs`` (the vLLM recipe's).
        preserve_kwarg: The template variable told whether earlier turns carry their reasoning
            (Qwen3.8: ``preserve_thinking``), or None to send no such variable.
        reasoning_field: The assistant-message field that carries reasoning back (vLLM and
            SGLang both read ``reasoning_content``).
        effort_notes: Per effort level, the instruction the chat template puts at the head of
            the system message while thinking is on. On a call whose own switch is off, the
            effort is not sent (the template would ignore it) and its note is put at the head of
            the system message, so the rendered prompt keeps the thinking turns' prefix.
        template_kwargs: Further ``chat_template_kwargs``; a key set here wins over the derived
            ``preserve_kwarg`` value.
        sampling: Further top-level sampling fields, for example ``{"top_p": 0.95}``.
        extra_body: Any other top-level request fields.
        tokenize_url: The server's tokenize endpoint for :meth:`OpenAIEngine.count_prompt_tokens`
            (vLLM: ``http://<host>:<port>/tokenize``, outside ``/v1``), or None: no counting.
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "RedHatAI/Qwen3.8-27B-INT4"
    api_key: str | None = None
    max_connections: int = 64
    connect_timeout_s: float = 10.0
    read_timeout_s: float | None = None
    pool_timeout_s: float | None = None
    max_retries: int = 5
    backoff_s: float = 1.0
    backoff_max_s: float = 30.0
    thinking: bool | None = None
    thinking_kwarg: str = "enable_thinking"
    effort_kwarg: str = "reasoning_effort"
    effort_transport: Literal["request", "template"] = "request"
    preserve_kwarg: str | None = "preserve_thinking"
    reasoning_field: str = "reasoning_content"
    effort_notes: Mapping[str, str] = field(default_factory=lambda: dict(QWEN38_EFFORT_NOTES))
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    sampling: Mapping[str, Any] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    tokenize_url: str | None = None

    def __post_init__(self) -> None:
        """Reject settings that cannot work.

        Raises:
            ValueError: If a size, timeout or retry setting is out of range, or the effort
                transport is unknown.
        """
        if self.max_connections < 1:
            raise ValueError(f"max_connections must be at least 1; got {self.max_connections}")
        optional = (self.read_timeout_s, self.pool_timeout_s)
        if self.connect_timeout_s <= 0 or any(t is not None and t <= 0 for t in optional):
            raise ValueError("timeouts must be positive (or None where allowed)")
        if self.max_retries < 0 or self.backoff_s < 0 or self.backoff_max_s < 0:
            raise ValueError("retry settings must be non-negative")
        if self.effort_transport not in ("request", "template"):
            raise ValueError(f"unknown effort_transport {self.effort_transport!r}")


def parse_completion(data: Any, seconds: float) -> Completion:
    """Convert an OpenAI chat completion response into a :class:`Completion`.

    Args:
        data: The parsed JSON response.
        seconds: The call's wall time.

    Returns:
        The completion: ``text`` from ``message.content`` (empty when null, as when the whole
        budget went to reasoning), ``reasoning`` from ``reasoning_content`` or ``reasoning``.
        Without either field, a content holding ``</think>`` is split on the last one: the
        reasoning before it (an opening ``<think>`` removed), the answer after it.

    Raises:
        EngineError: If the response has no choice, no message, or no token usage.
    """
    try:
        choice = data["choices"][0]
        message = choice["message"]
        usage = data["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        finish_reason = str(choice.get("finish_reason") or "unknown")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as err:
        raise EngineError(f"malformed chat completion: {err!r}") from err
    text = text if isinstance(text, str) else ""
    reasoning = reasoning if isinstance(reasoning, str) else None
    if reasoning is None and THINK_CLOSE in text:
        head, _, tail = text.rpartition(THINK_CLOSE)
        head = head.strip()
        reasoning = head.removeprefix(THINK_OPEN).strip()
        text = tail.lstrip()
    return Completion(
        text=text,
        reasoning=reasoning,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        seconds=seconds,
    )


class OpenAIEngine:
    """An :class:`~arc26.agent.types.Engine` backed by an OpenAI-compatible server.

    Use it as ``async with OpenAIEngine(config) as engine:``, or call :meth:`aclose` at the end.
    The HTTP client is created on the first call, inside the running event loop.
    """

    def __init__(
        self, config: EngineConfig | None = None, *, timing: TimingLogger | None = None
    ) -> None:
        """Create the engine; nothing connects until the first call.

        Args:
            config: The settings; the defaults when None.
            timing: Where each call's timing record goes, or None.
        """
        self.config = config if config is not None else EngineConfig()
        self._timing = timing
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> OpenAIEngine:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the connection pool (a pool of another, finished event loop is just dropped)."""
        client, loop = self._client, self._client_loop
        self._client = self._client_loop = None
        if client is not None and loop is asyncio.get_running_loop():
            await client.aclose()

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            config = self.config
            headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
            self._client_loop = loop
            self._client = httpx.AsyncClient(
                base_url=config.base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=config.connect_timeout_s,
                    read=config.read_timeout_s,
                    write=config.connect_timeout_s,
                    pool=config.pool_timeout_s,
                ),
                # No idle keep-alive: a call lasts seconds, so a new localhost connection per
                # call costs nothing measurable, and no idle socket outlives its event loop.
                limits=httpx.Limits(
                    max_connections=config.max_connections,
                    max_keepalive_connections=0,
                ),
            )
        return self._client

    def request_body(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
        thinking_token_budget: int | None = None,
    ) -> dict[str, Any]:
        """Build the JSON body of one chat completion request.

        Args:
            messages: The conversation.
            max_tokens: Cap on generated tokens, reasoning included.
            temperature: Sampling temperature.
            seed: Sampling seed, or None.
            reasoning_effort: The effort level, or None to send none.
            thinking: This call's thinking switch, or None for ``EngineConfig.thinking``. When
                it is False, the effort is not sent and its ``effort_notes`` entry heads the
                system message, as the template renders it with thinking on.
            thinking_token_budget: The reasoning tokens after which the server forces the end
                of thinking inside this request (vLLM's top-level ``thinking_token_budget``), or
                None to send none.

        Returns:
            The body: model, messages (an assistant message's reasoning in
            ``reasoning_field``), limits and sampling, the effort by the configured transport,
            and ``chat_template_kwargs`` when any is set
            (``preserve_kwarg`` is true when an earlier turn carries reasoning).
        """
        config = self.config
        sent = []
        carries_reasoning = False
        note = None
        if thinking is False and reasoning_effort is not None:
            note = config.effort_notes.get(reasoning_effort)
            reasoning_effort = None
        for k, m in enumerate(messages):
            item = {"role": m.role, "content": m.content}
            if k == 0 and m.role == "system" and note:
                item["content"] = f"{note}\n\n{m.content.strip()}"
            if m.role == "assistant" and m.reasoning is not None:
                item[config.reasoning_field] = m.reasoning
                carries_reasoning = True
            sent.append(item)
        body: dict[str, Any] = {
            "model": config.model,
            "messages": sent,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **config.sampling,
            **config.extra_body,
        }
        if seed is not None:
            body["seed"] = seed
        if thinking_token_budget is not None:
            body["thinking_token_budget"] = thinking_token_budget
        template = dict(config.template_kwargs)
        if config.preserve_kwarg:
            template.setdefault(config.preserve_kwarg, carries_reasoning)
        switch = config.thinking if thinking is None else thinking
        if switch is not None:
            template[config.thinking_kwarg] = switch
        if reasoning_effort is not None:
            if config.effort_transport == "template":
                template[config.effort_kwarg] = reasoning_effort
            else:
                body[config.effort_kwarg] = reasoning_effort
        if template:
            body["chat_template_kwargs"] = template
        return body

    def _backoff(self, retry: int) -> float:
        delay = min(self.config.backoff_max_s, self.config.backoff_s * 2**retry)
        return delay * random.uniform(0.5, 1.0)

    async def _post(self, body: Mapping[str, Any]) -> Any:
        client = self._http()
        retries = self.config.max_retries
        retry = 0
        while True:
            try:
                response = await client.post("chat/completions", json=body)
            except RETRYABLE as err:
                if retry == retries:
                    raise EngineError(f"no connection after {retries + 1} tries: {err!r}") from err
                delay = self._backoff(retry)
                retry += 1
                logger.warning("engine connection failed (%r); retry in %.2f s", err, delay)
                await asyncio.sleep(delay)
                continue
            except httpx.HTTPError as err:
                raise EngineError(f"request failed: {err!r}") from err
            if response.status_code >= 400:
                raise EngineError(f"HTTP {response.status_code}: {response.text[:500]}")
            try:
                return response.json()
            except ValueError as err:
                raise EngineError(f"response is not JSON: {response.text[:200]!r}") from err

    async def count_prompt_tokens(
        self,
        messages: Sequence[Message],
        *,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> int:
        """Count a request's prompt tokens as the server renders it (vLLM's ``/tokenize``).

        Why: the loop's first prompt bound counts one token per character, which over-counts
        prose about 4 times (run ``P6.3-02``, revision 4: bound 7,598 to 12,380 for prompts of
        2,494 to 7,291 tokens), and a phase 2 opening that shows earlier programs and their
        reports lost most of its first turn to it. The server renders the same messages and
        template variables with the served chat template, so its count is the prompt's.

        Args:
            messages: The conversation, as :meth:`complete` would send it.
            reasoning_effort: The effort the call would send (a template variable here, as vLLM
                forwards the top-level field into the template).
            thinking: The call's thinking switch, or None for ``EngineConfig.thinking``.

        Returns:
            The prompt's token count.

        Raises:
            EngineError: If no ``tokenize_url`` is set, or the request fails or its reply has no
                integer ``count``. No retry: the loop then uses its character bound.
        """
        config = self.config
        if not config.tokenize_url:
            raise EngineError("no tokenize_url: prompt counting is off")
        body = self.request_body(
            messages,
            max_tokens=1,
            temperature=1.0,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
        )
        template = dict(body.get("chat_template_kwargs") or {})
        if config.effort_kwarg in body:
            template[config.effort_kwarg] = body[config.effort_kwarg]
        request = {
            "model": config.model,
            "messages": body["messages"],
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": template,
        }
        try:
            response = await self._http().post(config.tokenize_url, json=request)
        except httpx.HTTPError as err:
            raise EngineError(f"tokenize failed: {err!r}") from err
        if response.status_code >= 400:
            raise EngineError(f"tokenize: HTTP {response.status_code}: {response.text[:300]}")
        try:
            count = response.json()["count"]
        except (ValueError, KeyError, TypeError) as err:
            raise EngineError(f"tokenize: malformed reply {response.text[:200]!r}") from err
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise EngineError(f"tokenize: count {count!r} is not a positive integer")
        return count

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
        thinking_token_budget: int | None = None,
    ) -> Completion:
        """Generate one reply.

        Args:
            messages: The conversation so far.
            max_tokens: Cap on generated tokens, reasoning included.
            temperature: Sampling temperature.
            seed: Sampling seed, or None.
            reasoning_effort: The effort level (Qwen3.8: ``"low"``, ``"medium"``, ``"xhigh"``),
                or None for the model's default. With ``thinking`` False it only picks the
                system message's effort note (``EngineConfig.effort_notes``).
            thinking: This call's thinking switch, or None for ``EngineConfig.thinking``.
            thinking_token_budget: The reasoning tokens after which the server ends the thinking
                and the answer follows in the same reply, or None (no budget sent).

        Returns:
            The reply, with the server's token counts and the call's wall time. With thinking
            off, a reply whose only text arrived as reasoning has it as its answer instead.

        Raises:
            EngineError: On an HTTP error, a timeout, a connection that fails after every
                retry, or a malformed response.
        """
        body = self.request_body(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            thinking_token_budget=thinking_token_budget,
        )
        start = time.perf_counter()
        data = await self._post(body)
        completion = parse_completion(data, time.perf_counter() - start)
        off = (self.config.thinking if thinking is None else thinking) is False
        if off and completion.reasoning and not completion.text.strip():
            completion = dataclasses.replace(completion, text=completion.reasoning, reasoning=None)
        if self._timing is not None:
            context = current_attempt()
            self._timing.record(
                None if context is None else context.task_id,
                ENGINE_OP,
                completion.seconds,
                completion.completion_tokens,
            )
        return completion
