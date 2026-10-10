"""Exercise the agent loop with a fake LLM client (no network required)."""

from __future__ import annotations

import asyncio

from agent_responder.agent import Agent
from agent_responder.llm import ChatMessage, LLMResponse, ToolCall
from agent_responder.tools import default_registry


class FakeClient:
    """A scripted client: first call emits an ``add`` tool call, the second
    returns a final answer that reflects the recorded tool result."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            call = ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})
            return LLMResponse(
                message=ChatMessage(role="assistant", content="", tool_calls=[call]),
                tool_calls=[call],
            )
        # Second call: a tool result must now be present in the message history.
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert tool_msgs, "expected the tool result to be in the message history"
        return LLMResponse(
            message=ChatMessage(role="assistant", content="The sum is 5."),
            tool_calls=[],
        )

    async def close(self) -> None:
        return None


def test_agent_calls_tool_then_returns_answer() -> None:
    agent = Agent(client=FakeClient(), tools=default_registry())
    answer = asyncio.run(agent.run("What is 2 + 3?"))

    assert answer == "The sum is 5."
    tool_msgs = [m for m in agent.conversation.messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].content == "5"


def test_agent_stops_at_step_budget() -> None:
    class AlwaysCalls:
        async def chat(self, messages, tools=None):
            call = ToolCall(id="c", name="echo", arguments={"text": "loop"})
            return LLMResponse(
                message=ChatMessage(role="assistant", content="", tool_calls=[call]),
                tool_calls=[call],
            )

        async def close(self) -> None:
            return None

    agent = Agent(client=AlwaysCalls(), tools=default_registry(), max_steps=3)
    answer = asyncio.run(agent.run("keep going"))
    assert answer.startswith("Stopped")
