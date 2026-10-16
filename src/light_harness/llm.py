"""LLM boundary: message/tool types, a client protocol, and a default
OpenAI-compatible implementation over ``httpx``.

Everything the :class:`Agent` needs is behind :class:`LLMClient`, so you can swap
in a different provider (or a fake for tests) without touching the loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from .config import Settings


class ToolCall(BaseModel):
    """A single function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the model's raw arguments could not be decoded; the call is not run
    # and this message is returned to the model as the tool result instead.
    error: str | None = None


class ChatMessage(BaseModel):
    """A single chat message in harness form.

    ``role`` is one of ``system`` / ``user`` / ``assistant`` / ``tool``. Tool
    results carry ``tool_call_id``; assistant messages that request tool calls
    carry ``tool_calls``.
    """

    role: str
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMResponse(BaseModel):
    """The model's reply: a message plus any requested tool calls."""

    message: ChatMessage
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMClient(Protocol):
    """Anything the agent can call to talk to a model."""

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send ``messages`` (plus optional tool schemas) and return a response.
        With ``on_token``, ask for a streamed reply and call it with each text
        delta as it arrives."""
        ...

    async def close(self) -> None:
        """Release underlying resources (e.g. the HTTP client)."""
        ...


def _to_wire(message: ChatMessage) -> dict[str, Any]:
    """Convert a harness message into an OpenAI-compatible chat message."""
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in message.tool_calls
            ],
        }
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    return {"role": message.role, "content": message.content}


_MAX_ATTEMPTS = 3
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Decode a tool call's arguments into ``(arguments, error)``.

    Models sometimes emit malformed JSON; that must reach the model as a tool error
    (so it can retry) rather than crash the run, so a failure yields ``({}, message)``.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            return {}, f"arguments are not valid JSON ({exc})"
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, f"arguments must be a JSON object, got {type(raw).__name__}"
    return raw, None


class LLMError(Exception):
    """The endpoint answered, but not with a usable chat completion."""


def _first_message(response: httpx.Response) -> dict[str, Any]:
    """Return ``choices[0].message`` from a 200 response, or raise :class:`LLMError`
    saying what was wrong with it (empty ``choices`` is common when a guardrail
    blocks a request)."""
    try:
        data = response.json()
    except ValueError:
        raise LLMError("the response body is not valid JSON") from None
    if not isinstance(data, dict):
        raise LLMError(f"the response is a JSON {type(data).__name__}, not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        detail = data.get("error")
        suffix = f": {detail}" if detail else ""
        raise LLMError(f"the response has no choices{suffix}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMError("the first choice has no message object")
    return message


def _tool_call_fields(raw: Any) -> tuple[str, str, Any]:
    """Return ``(id, name, arguments)`` of a wire tool call, or raise
    :class:`LLMError`. Without an id or name the call cannot be answered, so it
    cannot be passed on to the model as a tool error either."""
    function = raw.get("function") if isinstance(raw, dict) else None
    if not isinstance(function, dict):
        raise LLMError("the response has a malformed tool call (no function object)")
    call_id = raw.get("id")
    name = function.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise LLMError("the response has a tool call without a string id and function name")
    return call_id, name, function.get("arguments")


class OpenAICompatibleClient:
    """Default client: posts to ``{base_url}/chat/completions``."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        retry_delay: float = 0.5,
    ) -> None:
        self._settings = settings
        self._retry_delay = retry_delay
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retry(
        self, payload: dict[str, Any], stream: bool = False
    ) -> httpx.Response:
        """POST the chat request, retrying transient failures (connection errors,
        429 and 5xx) with exponential backoff; other HTTP errors raise at once.
        With ``stream`` the body is left unread and the caller must close it."""
        for attempt in range(_MAX_ATTEMPTS):
            last = attempt == _MAX_ATTEMPTS - 1
            try:
                request = self._client.build_request("POST", "/chat/completions", json=payload)
                response = await self._client.send(request, stream=stream)
            except httpx.TransportError:
                if last:
                    raise
            else:
                if response.status_code not in _RETRY_STATUSES or last:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError:
                        await response.aclose()
                        raise
                    return response
                await response.aclose()
            await asyncio.sleep(self._retry_delay * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover - loop always returns/raises

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [_to_wire(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if on_token is not None:
            payload["stream"] = True
            content, raw_calls = await self._read_stream(payload, on_token)
        else:
            choice = _first_message(await self._post_with_retry(payload))
            content = choice.get("content") or ""
            raw_calls = choice.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for raw in raw_calls:
            call_id, name, arguments = _tool_call_fields(raw)
            args, error = _parse_tool_arguments(arguments)
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=args, error=error))

        return LLMResponse(
            message=ChatMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
            ),
            tool_calls=tool_calls,
        )

    async def _read_stream(
        self, payload: dict[str, Any], on_token: Callable[[str], None]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Consume an SSE chat completion, calling ``on_token`` for each content
        delta. Returns the full content and the reassembled wire tool calls."""
        response = await self._post_with_retry(payload, stream=True)
        parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    raise LLMError("the stream contains an invalid JSON event") from None
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise LLMError(f"the stream reported an error: {chunk['error']}")
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if text:
                    parts.append(text)
                    on_token(text)
                for raw in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        raw.get("index", len(calls)),
                        {"id": None, "function": {"name": "", "arguments": ""}},
                    )
                    if raw.get("id"):
                        slot["id"] = raw["id"]
                    function = raw.get("function") or {}
                    if function.get("name"):
                        slot["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]
        finally:
            await response.aclose()
        if not parts and not calls:
            raise LLMError("the stream produced no content or tool calls")
        return "".join(parts), [calls[i] for i in sorted(calls)]
