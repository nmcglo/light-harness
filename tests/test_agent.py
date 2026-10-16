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


def _chat_once(handler):
    client, http = _mock_client(handler)
    try:
        return asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
    finally:
        asyncio.run(http.aclose())


def test_unusable_responses_raise_llm_error_naming_the_problem() -> None:
    import httpx
    import pytest

    from light_harness.llm import LLMError

    bad_call = {"type": "function", "function": {"name": "add", "arguments": "{}"}}  # no id
    cases = {
        "no choices": (httpx.Response(200, json={"choices": []}), "no choices"),
        "guardrail error": (
            httpx.Response(200, json={"error": "blocked by policy"}),
            "blocked by policy",
        ),
        "not json": (httpx.Response(200, content=b"<html>gateway</html>"), "not valid JSON"),
        "no message": (httpx.Response(200, json={"choices": [{}]}), "no message"),
        "tool call without id": (
            httpx.Response(200, json=_completion({"content": "", "tool_calls": [bad_call]})),
            "without a string id",
        ),
        "tool call not an object": (
            httpx.Response(200, json=_completion({"content": "", "tool_calls": ["x"]})),
            "malformed tool call",
        ),
    }
    for name, (response, expected) in cases.items():
        with pytest.raises(LLMError, match=expected):
            _chat_once(lambda request, r=response: r)
        assert name  # keeps the failing case visible in the traceback


def test_transport_errors_are_retried_with_exponential_backoff(monkeypatch) -> None:
    import httpx
    import pytest

    from light_harness.config import Settings
    from light_harness.llm import OpenAICompatibleClient

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("light_harness.llm.asyncio.sleep", fake_sleep)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=_completion({"content": "ok"}))

    def build(h) -> tuple[OpenAICompatibleClient, httpx.AsyncClient]:
        http = httpx.AsyncClient(base_url="http://llm.test/v1", transport=httpx.MockTransport(h))
        return OpenAICompatibleClient(Settings(), client=http, retry_delay=1.0), http

    client, http = build(handler)
    try:
        response = asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
    finally:
        asyncio.run(http.aclose())
    assert response.message.content == "ok"
    assert sleeps == [1.0, 2.0]

    def always_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    sleeps.clear()
    client, http = build(always_down)
    try:
        with pytest.raises(httpx.ConnectError):
            asyncio.run(client.chat([ChatMessage(role="user", content="hi")]))
    finally:
        asyncio.run(http.aclose())
    assert sleeps == [1.0, 2.0]  # no sleep after the final attempt


def test_repl_survives_a_failed_turn(monkeypatch) -> None:
    from light_harness import cli
    from light_harness.llm import LLMError

    class FlakyAgent:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run(self, prompt: str, on_token=None) -> str:
            self.prompts.append(prompt)
            if prompt == "boom":
                raise LLMError("the response has no choices")
            on_token(f"answer to {prompt}")
            return f"answer to {prompt}"

    lines = iter(["boom", "again", "/exit"])
    monkeypatch.setattr(cli.console, "input", lambda *_a, **_k: next(lines))
    printed: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda *a, **_k: printed.append(str(a[0])))
    errors: list[str] = []
    monkeypatch.setattr(cli.err_console, "print", lambda *a, **_k: errors.append(str(a[0])))
    agent = FlakyAgent()

    asyncio.run(cli._run_repl(agent))  # type: ignore[arg-type]

    assert agent.prompts == ["boom", "again"]
    assert "answer to again" in printed
    assert any("no choices" in e for e in errors)


def test_streamed_chat_reports_tokens_and_reassembles_tool_calls() -> None:
    import json

    import httpx

    from light_harness.config import Settings
    from light_harness.llm import ChatMessage, OpenAICompatibleClient

    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
            "function": {"name": "add", "arguments": '{"a": 1,'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
            "function": {"arguments": ' "b": 2}'}}]}}]},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http = httpx.AsyncClient(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(Settings(), client=http, retry_delay=0)
    tokens: list[str] = []
    try:
        response = asyncio.run(
            client.chat([ChatMessage(role="user", content="hi")], on_token=tokens.append)
        )
    finally:
        asyncio.run(http.aclose())

    assert seen[0]["stream"] is True
    assert tokens == ["Hel", "lo"]
    assert response.message.content == "Hello"
    assert response.tool_calls[0].name == "add"
    assert response.tool_calls[0].arguments == {"a": 1, "b": 2}
