"""LLM boundary: message/tool types, a client protocol, and a default
OpenAI-compatible implementation over ``httpx``.

Everything the :class:`Agent` needs is behind :class:`LLMClient`, so you can swap
in a different provider (or a fake for tests) without touching the loop.
"""

from __future__ import annotations

import asyncio
import json
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
    ) -> LLMResponse:
        """Send ``messages`` (plus optional tool schemas) and return a response."""
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

    async def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        """POST the chat request, retrying transient failures (connection errors,
        429 and 5xx) with exponential backoff; other HTTP errors raise at once."""
        for attempt in range(_MAX_ATTEMPTS):
            last = attempt == _MAX_ATTEMPTS - 1
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError:
                if last:
                    raise
            else:
                if response.status_code not in _RETRY_STATUSES or last:
                    response.raise_for_status()
                    return response
            await asyncio.sleep(self._retry_delay * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover - loop always returns/raises

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [_to_wire(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = await self._post_with_retry(payload)
        choice = response.json()["choices"][0]["message"]

        tool_calls: list[ToolCall] = []
        for raw in choice.get("tool_calls") or []:
            args, error = _parse_tool_arguments(raw["function"]["arguments"])
            tool_calls.append(
                ToolCall(
                    id=raw["id"],
                    name=raw["function"]["name"],
                    arguments=args,
                    error=error,
                )
            )

        return LLMResponse(
            message=ChatMessage(
                role="assistant",
                content=choice.get("content") or "",
                tool_calls=tool_calls or None,
            ),
            tool_calls=tool_calls,
        )
