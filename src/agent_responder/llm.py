"""LLM boundary: message/tool types, a client protocol, and a default
OpenAI-compatible implementation over ``httpx``.

Everything the :class:`Agent` needs is behind :class:`LLMClient`, so you can swap
in a different provider (or a fake for tests) without touching the loop.
"""

from __future__ import annotations

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


class OpenAICompatibleClient:
    """Default client: posts to ``{base_url}/chat/completions``."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

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

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]["message"]

        tool_calls: list[ToolCall] = []
        for raw in choice.get("tool_calls") or []:
            args = raw["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            tool_calls.append(
                ToolCall(id=raw["id"], name=raw["function"]["name"], arguments=args or {})
            )

        return LLMResponse(
            message=ChatMessage(
                role="assistant",
                content=choice.get("content") or "",
                tool_calls=tool_calls or None,
            ),
            tool_calls=tool_calls,
        )
