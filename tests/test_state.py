"""Conversation history helpers and ``Agent.reset``, the public API for code that
embeds an ``Agent`` and manages its history directly."""

from __future__ import annotations

import asyncio

from light_harness.agent import Agent
from light_harness.llm import ChatMessage, LLMResponse, ToolCall
from light_harness.state import Conversation
from light_harness.tools import default_registry


def test_history_helpers_build_a_well_formed_tool_call_exchange() -> None:
    conversation = Conversation()
    call = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})

    conversation.add_user("what is 1 + 2?")
    conversation.add_assistant("", tool_calls=[call])
    conversation.add_tool("c1", "3")
    conversation.add_assistant("It is 3.")

    assert [m.role for m in conversation.messages] == ["user", "assistant", "tool", "assistant"]
    assert conversation.messages[0].content == "what is 1 + 2?"
    assert conversation.messages[1].tool_calls == [call]
    assert conversation.messages[2].tool_call_id == "c1"  # ties the result to the call
    assert conversation.messages[2].content == "3"
    assert conversation.messages[3].tool_calls is None


def test_reset_empties_the_history() -> None:
    conversation = Conversation()
    conversation.add_user("hello")
    conversation.add_assistant("hi")

    conversation.reset()

    assert conversation.messages == []
    conversation.add_user("fresh")  # still usable afterwards
    assert [m.content for m in conversation.messages] == ["fresh"]


def test_agent_reset_starts_a_new_session_for_the_model() -> None:
    """After ``reset`` the model must not see the earlier turns again."""
    seen: list[list[str]] = []

    class RecordingClient:
        async def chat(
            self, messages: list[ChatMessage], tools: list[dict] | None = None
        ) -> LLMResponse:
            seen.append([m.content for m in messages if m.role == "user"])
            return LLMResponse(message=ChatMessage(role="assistant", content="ok"))

        async def close(self) -> None:
            return None

    agent = Agent(RecordingClient(), default_registry())

    asyncio.run(agent.run("first"))
    asyncio.run(agent.run("second"))
    agent.reset()
    asyncio.run(agent.run("third"))

    assert seen == [["first"], ["first", "second"], ["third"]]
