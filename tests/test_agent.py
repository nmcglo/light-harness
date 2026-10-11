"""Exercise the agent loop with a fake LLM client (no network required)."""

from __future__ import annotations

import asyncio

from light_harness.agent import Agent
from light_harness.llm import ChatMessage, LLMResponse, ToolCall
from light_harness.tools import default_registry


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


def test_tool_call_times_out_and_reports_error() -> None:
    import asyncio

    from light_harness.tools import ToolRegistry

    class SlowTool:
        name = "slow"
        description = "never returns in time"
        parameters: dict = {"type": "object", "properties": {}}

        async def run(self, arguments: dict) -> str:
            await asyncio.sleep(5)
            return "done"

    registry = ToolRegistry(timeout=0.05)
    registry.register(SlowTool())
    result = asyncio.run(registry.run("slow", {}))
    assert result == "Error: tool 'slow' timed out after 0.05s."


def test_registry_validates_arguments_before_running() -> None:
    registry = default_registry()

    async def run(name: str, args: dict) -> str:
        return await registry.run(name, args)

    assert asyncio.run(run("echo", {})) == (
        "Error: invalid arguments for 'echo': missing required argument 'text'."
    )
    # "1" + "2" would otherwise concatenate to "12".
    result = asyncio.run(run("add", {"a": "1", "b": "2"}))
    assert result == (
        "Error: invalid arguments for 'add': "
        "argument 'a' must be of type number, got str."
    )
    assert "must be of type number, got bool" in asyncio.run(run("add", {"a": True, "b": 1}))
    assert asyncio.run(run("add", {"a": 1, "b": 2.5})) == "3.5"
    # null for an optional argument is treated as absent.
    assert "Error" not in asyncio.run(run("current_time", {"x": None}))


def _mock_client(handler):
    import httpx

    from light_harness.config import Settings
    from light_harness.llm import OpenAICompatibleClient

    http = httpx.AsyncClient(
        base_url="http://llm.test/v1", transport=httpx.MockTransport(handler)
    )
    return OpenAICompatibleClient(Settings(), client=http, retry_delay=0), http


def _completion(message: dict) -> dict:
    return {"choices": [{"message": message}]}


def test_malformed_tool_arguments_reach_the_model_instead_of_crashing() -> None:
    import httpx

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:
            bad_call = {
                "id": "c1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": '},
            }
            return httpx.Response(200, json=_completion({"content": "", "tool_calls": [bad_call]}))
        return httpx.Response(200, json=_completion({"content": "recovered"}))

    client, http = _mock_client(handler)
    agent = Agent(client=client, tools=default_registry())
    try:
        assert asyncio.run(agent.run("add please")) == "recovered"
    finally:
        asyncio.run(http.aclose())
    tool_msg = next(m for m in sent[1]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"].startswith(
        "Error: invalid arguments for 'add': arguments are not valid JSON"
    )


def test_llm_client_retries_transient_errors_but_not_client_errors() -> None:
    import httpx

    statuses = iter([503, 429, 200])
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        status = next(statuses)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json=_completion({"content": "ok"}))

    client, http = _mock_client(flaky)
    try:
        response = asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
        assert response.message.content == "ok"
        assert calls["n"] == 3

        calls["n"] = 0

        def unauthorized(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401)

        client, http2 = _mock_client(unauthorized)
        try:
            import pytest

            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
        finally:
            asyncio.run(http2.aclose())
        assert calls["n"] == 1  # 401 is not retried

        calls["n"] = 0

        def always_down(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500)

        client, http3 = _mock_client(always_down)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
        finally:
            asyncio.run(http3.aclose())
        assert calls["n"] == 3  # gives up after the attempt budget
    finally:
        asyncio.run(http.aclose())
