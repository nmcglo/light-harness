"""Multi-turn conversation state.

The agent keeps a single :class:`Conversation` as the source of truth for the
message history, so repeated ``run`` calls stay coherent across turns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import ChatMessage, ToolCall


class Conversation(BaseModel):
    """An ordered list of chat messages for a single session.

    The agent loop itself only uses :meth:`add`. ``add_user``, ``add_assistant``,
    ``add_tool`` and ``reset`` (and ``Agent.reset``) are public conveniences for
    code that embeds :class:`~light_harness.agent.Agent` and drives the history
    directly; they are covered by ``tests/test_state.py``."""

    messages: list[ChatMessage] = Field(default_factory=list)

    def add(self, message: ChatMessage) -> None:
        """Append a message to the history."""
        self.messages.append(message)

    def add_user(self, content: str) -> None:
        self.add(ChatMessage(role="user", content=content))

    def add_assistant(
        self, content: str, tool_calls: list[ToolCall] | None = None
    ) -> None:
        self.add(ChatMessage(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool(self, tool_call_id: str, content: str) -> None:
        self.add(ChatMessage(role="tool", tool_call_id=tool_call_id, content=content))

    def reset(self) -> None:
        self.messages.clear()
