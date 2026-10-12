"""Tests for webhook listeners (webhooks.py).

The server tests drive a real ``ThreadingHTTPServer`` on an ephemeral port with
a capturing fake LLM client, so routing, bearer auth, body parsing, and the
agent round-trip are exercised end to end without any network.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from light_harness.agent import Agent
from light_harness.config import AgentConfig, Config, Interaction, WebhookConfig
from light_harness.llm import ChatMessage, LLMResponse
from light_harness.tools import build_tool_registry
from light_harness.webhooks import (
    WebhookHandler,
    _bearer_matches,
    create_webhook_server,
    parse_webhook_body,
    tool_availability_warning,
    unauthenticated_warning,
)


def _write_cfg(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")


class CapturingClient:
    """A fake LLM client: records user prompts and the system prompt; fixed answer."""

    def __init__(self, answer: str = "handled") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def chat(self, messages, tools=None) -> LLMResponse:
        for message in messages:
            if message.role == "system":
                self.system_prompts.append(message.content)
            elif message.role == "user":
                self.prompts.append(message.content)
        return LLMResponse(
            message=ChatMessage(role="assistant", content=self.answer), tool_calls=[]
        )

    async def close(self) -> None:
        return None


@pytest.fixture()
def webhook_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A webhook server on an ephemeral port backed by a capturing fake client."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        agents:
          - name: assistant
            provider: local
            model: m
            tools: []
        webhooks:
          - path: /hooks/open
            agent: assistant
          - path: /hooks/secret
            agent: assistant
            secret: topsecret
          - path: /hooks/custom
            agent: assistant
            system_prompt: You are a deploy bot.
        """,
    )
    config = Config.load()
    client = CapturingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        return Agent(
            client,
            build_tool_registry(agent_cfg.tools),
            system_prompt=webhook.system_prompt or "sp",
        )

    server = create_webhook_server(config, "127.0.0.1", 0, agent_factory=factory)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", client
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _post(url: str, data: bytes, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _wait_for_prompts(client: CapturingClient, n: int = 1, timeout: float = 5.0) -> None:
    """Wait (the agent runs in a background thread) until n prompts were seen."""
    deadline = time.monotonic() + timeout
    while len(client.prompts) < n:
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {n} prompt(s): {client.prompts}")
        time.sleep(0.01)


def test_parse_webhook_body_json() -> None:
    assert parse_webhook_body(b'{"a": 1, "b": [2, 3]}') == (
        '{\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}'
    )


def test_parse_webhook_body_plain_text() -> None:
    assert parse_webhook_body(b"hello world") == "hello world"


def test_parse_webhook_body_empty() -> None:
    assert parse_webhook_body(b"") == ""
    assert parse_webhook_body(b"   \n\t ") == ""


def test_bearer_matches_rules() -> None:
    assert _bearer_matches(None, None) is True
    assert _bearer_matches(None, "Bearer whatever") is True
    assert _bearer_matches("s3cret", "Bearer s3cret") is True
    assert _bearer_matches("s3cret", "Bearer wrong") is False
    assert _bearer_matches("s3cret", None) is False
    assert _bearer_matches("s3cret", "Basic abc") is False
    assert _bearer_matches("s3cret", "s3cret") is False  # no Bearer scheme


def _config_with(
    *hooks: WebhookConfig, agents: list[AgentConfig] | None = None
) -> Config:
    return Config({}, agents or [], {}, Interaction(), list(hooks))


def test_unauthenticated_warning_counts_open_webhooks() -> None:
    config = _config_with(
        WebhookConfig(path="/a", secret="s"),
        WebhookConfig(path="/b"),
        WebhookConfig(path="/c", secret=""),
    )
    warning = unauthenticated_warning(config)
    assert warning is not None
    assert "2 of 3" in warning
    assert "/b" in warning and "/c" in warning and "/a" not in warning


def test_no_warning_when_all_webhooks_authenticated() -> None:
    assert unauthenticated_warning(_config_with(WebhookConfig(path="/a", secret="s"))) is None
    assert unauthenticated_warning(_config_with()) is None


def test_tool_availability_warning_names_implied_file_tools_without_a_workspace() -> None:
    config = _config_with(
        WebhookConfig(path="/hooks/write", agent="writer"),
        agents=[AgentConfig(name="writer")],
    )
    warning = tool_availability_warning(config)
    assert warning is not None
    # No "tools" list means every built-in tool, and the file tools need a workspace.
    assert "/hooks/write" in warning
    assert "read_file, write_file, edit_file" in warning


def test_tool_availability_warning_is_quiet_when_nothing_implied_is_lost() -> None:
    config = _config_with(
        WebhookConfig(path="/a", agent="with_ws"),
        WebhookConfig(path="/b", agent="explicit"),
        agents=[
            AgentConfig(name="with_ws", workspace="/tmp/ws"),
            AgentConfig(name="explicit", tools=["echo"]),
        ],
    )
    assert tool_availability_warning(config) is None


def test_tool_availability_warning_names_an_undefined_agent() -> None:
    config = _config_with(
        WebhookConfig(path="/h", agent="ghost"), agents=[AgentConfig(name="present")]
    )
    warning = tool_availability_warning(config)
    assert warning is not None
    assert "/h" in warning and "ghost" in warning


def test_config_load_parses_webhooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        agents:
          - name: assistant
            provider: local
            model: m
        webhooks:
          - path: /hooks/deploy
            agent: assistant
            secret: tok
          - path: /hooks/plain
            agent: assistant
          - path: /hooks/custom
            agent: assistant
            system_prompt: Custom prompt here
          - path: /hooks/emptyprompt
            agent: assistant
            system_prompt: "   "
        """,
    )
    config = Config.load()
    assert [w.path for w in config.webhooks] == [
        "/hooks/deploy",
        "/hooks/plain",
        "/hooks/custom",
        "/hooks/emptyprompt",
    ]
    deploy = config.webhook_by_path("/hooks/deploy")
    assert deploy is not None and deploy.secret == "tok"
    plain = config.webhook_by_path("/hooks/plain")
    assert plain is not None and plain.secret is None
    custom = config.webhook_by_path("/hooks/custom")
    assert custom is not None and custom.system_prompt == "Custom prompt here"
    empty = config.webhook_by_path("/hooks/emptyprompt")
    assert empty is not None and empty.system_prompt is None
    assert config.webhook_by_path("/nope") is None
    assert config.get_agent("assistant") is not None


def test_parse_webhooks_drops_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        webhooks:
          - path: /hooks/ok
            agent: assistant
          - path: /hooks/sp
            agent: assistant
            system_prompt: Custom
          - path: /hooks/badsp
            agent: assistant
            system_prompt: [1, 2]
          - path: not-a-path
            agent: assistant
          - agent: assistant
          - path: /hooks/ok
            agent: assistant
        """,
    )
    config = Config.load()
    assert [w.path for w in config.webhooks] == ["/hooks/ok", "/hooks/sp"]
    assert config.webhook_by_path("/hooks/sp").system_prompt == "Custom"


def test_endpoint_for_scopes_to_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        providers:
          local:
            base_url: http://x/v1
            api_key: k1
        agents:
          - name: a
            provider: local
            model: ma
        """,
    )
    config = Config.load()
    agent = config.get_agent("a")
    assert agent is not None
    endpoint = config.endpoint_for(agent)
    assert endpoint.model == "ma"
    assert endpoint.base_url == "http://x/v1"


def test_webhook_runs_agent_with_parsed_body(webhook_server) -> None:
    base, client = webhook_server
    status, data = _post(base + "/hooks/open", b'{"event": "deploy", "n": 1}')
    assert status == 200
    assert data == {"agent": "assistant", "message": "OK"}
    _wait_for_prompts(client)
    assert client.prompts == ['{\n  "event": "deploy",\n  "n": 1\n}']


def test_webhook_no_body_sends_empty_prompt(webhook_server) -> None:
    base, client = webhook_server
    status, data = _post(base + "/hooks/open", b"")
    assert status == 200
    assert data["message"] == "OK"
    _wait_for_prompts(client)
    assert client.prompts == [""]


def test_webhook_requires_bearer_secret(webhook_server) -> None:
    base, client = webhook_server
    assert _post(base + "/hooks/secret", b'{"a": 1}')[0] == 401
    assert _post(
        base + "/hooks/secret", b'{"a": 1}', headers={"Authorization": "Bearer nope"}
    )[0] == 401
    status, data = _post(
        base + "/hooks/secret",
        b'{"a": 1}',
        headers={"Authorization": "Bearer topsecret"},
    )
    assert status == 200
    assert data == {"agent": "assistant", "message": "OK"}
    _wait_for_prompts(client)
    assert client.prompts == ['{\n  "a": 1\n}']


def test_webhook_unknown_path_is_404(webhook_server) -> None:
    base, _ = webhook_server
    status, data = _post(base + "/hooks/missing", b"{}")
    assert status == 404
    assert "message" in data


def test_webhook_unknown_agent_is_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A webhook referencing an unknown agent is an error, not an ack."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        agents:
          - name: assistant
            provider: local
            model: m
        webhooks:
          - path: /hooks/bad
            agent: ghost
        """,
    )
    config = Config.load()
    server = create_webhook_server(config, "127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _post(f"http://127.0.0.1:{port}/hooks/bad", b"{}")
        assert status == 500
        assert "message" in data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_webhook_get_is_405(webhook_server) -> None:
    base, _ = webhook_server
    status, _ = _get(base + "/hooks/open")
    assert status == 405


def test_webhook_custom_system_prompt_reaches_agent(webhook_server) -> None:
    base, client = webhook_server
    status, _ = _post(base + "/hooks/custom", b'{"a": 1}')
    assert status == 200
    _wait_for_prompts(client)
    assert client.system_prompts == ["You are a deploy bot."]
    assert len(client.prompts) == 1


def test_webhook_default_system_prompt_when_unset(webhook_server) -> None:
    base, client = webhook_server
    status, _ = _post(base + "/hooks/open", b'{"a": 1}')
    assert status == 200
    _wait_for_prompts(client)
    assert client.system_prompts == ["sp"]


def _connect(url: str) -> socket.socket:
    """A raw connection to the test server."""
    host, port = url.removeprefix("http://").split(":")
    return socket.create_connection((host, int(port)), timeout=5)


def _read_response(stream: io.BufferedReader) -> tuple[int, dict[str, str]]:
    """The next complete response as (status, lowercased headers); (0, {}) on EOF,
    which is how a server that closed the connection looks to the client."""
    first = stream.readline()
    if not first:
        return 0, {}
    status = int(first.decode().split(" ", 2)[1])
    headers = {}
    while (line := stream.readline()) not in (b"\r\n", b"\n", b""):
        name, _, value = line.decode("iso-8859-1").partition(":")
        headers[name.lower()] = value.strip()
    # The body is read too, so the next response on the connection starts in the right place.
    stream.read(int(headers.get("content-length", "0")))
    return status, headers


def _raw_request(url: str, request: bytes) -> tuple[int, dict[str, str]]:
    """Send raw request bytes on one connection and read its response."""
    with _connect(url) as s:
        stream = s.makefile("rb")
        s.sendall(request)
        return _read_response(stream)


def _chunked_request(
    chunks: list[str], path: str = "/hooks/open", extra: dict[str, str] | None = None
) -> bytes:
    """Raw request bytes for a chunked POST; ``chunks`` are the chunk payloads."""
    lines = [f"POST {path} HTTP/1.1", "Host: x", "Transfer-Encoding: chunked"]
    lines += [f"{name}: {value}" for name, value in (extra or {}).items()]
    request = "\r\n".join(lines) + "\r\n\r\n"
    for chunk in chunks:
        request += f"{len(chunk):x}\r\n{chunk}\r\n"
    return (request + "0\r\n\r\n").encode()


def test_webhook_rejects_oversized_body_without_reading_it(webhook_server) -> None:
    url, client = webhook_server
    status, headers = _raw_request(
        url, b"POST /hooks/open HTTP/1.1\r\nHost: x\r\nContent-Length: 10485760\r\n\r\n"
    )
    assert status == 413
    # The oversized body is left on the socket, so the connection cannot be reused.
    assert headers["connection"] == "close"
    assert client.prompts == []


@pytest.mark.parametrize("value", ["abc", "-5"])
def test_webhook_rejects_malformed_content_length(webhook_server, value: str) -> None:
    url, client = webhook_server
    status, headers = _raw_request(
        url, f"POST /hooks/open HTTP/1.1\r\nHost: x\r\nContent-Length: {value}\r\n\r\n".encode()
    )
    assert status == 400
    assert headers["connection"] == "close"
    assert client.prompts == []


def test_chunked_body_reaches_the_agent(webhook_server) -> None:
    url, client = webhook_server
    assert _raw_request(url, _chunked_request(['{"a":', " 1}"]))[0] == 200
    _wait_for_prompts(client)
    assert client.prompts == [json.dumps({"a": 1}, indent=2)]


def test_chunked_outranks_content_length(webhook_server) -> None:
    """RFC 9112 3.3.3: the chunked framing is what defines the message length."""
    url, client = webhook_server
    status, _ = _raw_request(url, _chunked_request(["hello"], extra={"Content-Length": "3"}))
    assert status == 200
    _wait_for_prompts(client)
    assert client.prompts == ["hello"]


def test_chunk_size_extensions_and_trailers_are_tolerated(webhook_server) -> None:
    url, client = webhook_server
    request = (
        b"POST /hooks/open HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        b"5;name=v\r\nhello\r\n0\r\nX-Trailer: yes\r\n\r\n"
    )
    assert _raw_request(url, request)[0] == 200
    _wait_for_prompts(client)
    assert client.prompts == ["hello"]


def test_chunked_body_over_the_limit_is_refused_without_running(webhook_server) -> None:
    """The oversized chunk is announced but not sent: the size check is what refuses
    it, and the bytes left on the socket force the close."""
    url, client = webhook_server
    request = (
        "POST /hooks/open HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        f"{1024 * 1024 + 1:x}\r\n"
    ).encode()
    status, headers = _raw_request(url, request)
    assert status == 413
    assert headers["connection"] == "close"
    assert client.prompts == []


@pytest.mark.parametrize("size_field", ["zz", "-1"])
def test_malformed_chunk_size_is_refused(webhook_server, size_field: str) -> None:
    url, client = webhook_server
    request = (
        "POST /hooks/open HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
        f"{size_field}\r\nhello\r\n"
    ).encode()
    status, headers = _raw_request(url, request)
    assert status == 400
    assert headers["connection"] == "close"
    assert client.prompts == []


def test_transfer_encoding_other_than_chunked_is_refused(webhook_server) -> None:
    url, client = webhook_server
    request = (
        b"POST /hooks/open HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip\r\n"
        b"Content-Length: 5\r\n\r\nhello"
    )
    status, headers = _raw_request(url, request)
    assert status == 400
    assert headers["connection"] == "close"
    assert client.prompts == []


def test_chunked_body_is_consumed_so_the_connection_can_be_reused(
    webhook_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body must be read to its end: otherwise the follow-up request on the same
    connection gets the leftover chunk bytes as its own request line."""
    monkeypatch.setattr(WebhookHandler, "protocol_version", "HTTP/1.1")  # keep-alive
    url, client = webhook_server
    follow_up = b"POST /hooks/open HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi"
    with _connect(url) as s:
        stream = s.makefile("rb")
        s.sendall(_chunked_request(["hello"]) + follow_up)
        assert _read_response(stream)[0] == 200
        assert _read_response(stream)[0] == 200
    _wait_for_prompts(client, 2)
    assert sorted(client.prompts) == ["hello", "hi"]


def test_body_left_unread_closes_the_connection(
    webhook_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 404 is answered before the body is read, so the connection must close
    instead of reading those bytes as the next request."""
    monkeypatch.setattr(WebhookHandler, "protocol_version", "HTTP/1.1")  # keep-alive
    url, client = webhook_server
    with _connect(url) as s:
        stream = s.makefile("rb")
        s.sendall(
            b"POST /hooks/nope HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello"
            b"POST /hooks/open HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi"
        )
        status, headers = _read_response(stream)
        assert status == 404
        assert headers["connection"] == "close"
        assert _read_response(stream) == (0, {})  # closed, so "hello" is never a request
    assert client.prompts == []


def test_webhook_limits_concurrent_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        """
        agents:
          - name: assistant
            tools: []
        webhooks:
          - path: /hooks/open
            agent: assistant
        """,
    )
    gate = threading.Event()

    class BlockingClient(CapturingClient):
        async def chat(self, messages, tools=None) -> LLMResponse:
            await asyncio.get_running_loop().run_in_executor(None, gate.wait, 5)
            return await super().chat(messages, tools)

    client = BlockingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        return Agent(client, build_tool_registry(agent_cfg.tools))

    server = create_webhook_server(
        Config.load(), "127.0.0.1", 0, agent_factory=factory, max_concurrent_runs=1
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/hooks/open"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _post(url, b"one")[0] == 200
        assert _post(url, b"two")[0] == 429  # slot held by the first, still-running run
        gate.set()
        _wait_for_prompts(client, 1)
        deadline = time.monotonic() + 5
        while _post(url, b"three")[0] == 429:  # slot is freed once the run finishes
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "GET"])
def test_non_post_methods_get_405_with_allow(webhook_server, method: str) -> None:
    import http.client

    url, client = webhook_server
    conn = http.client.HTTPConnection(url.removeprefix("http://"), timeout=5)
    try:
        conn.request(method, "/hooks/open")
        response = conn.getresponse()
        body = response.read()
        assert response.status == 405
        assert response.getheader("Allow") == "POST"
        if method == "HEAD":
            assert body == b""  # headers only
        else:
            assert json.loads(body)["message"] == "webhooks accept POST only"
    finally:
        conn.close()
    assert client.prompts == []


def test_release_agent_closes_session_even_if_client_close_fails(tmp_path: Path) -> None:
    from light_harness.sessions import SessionLog
    from light_harness.webhooks import _release_agent

    class BrokenClient(CapturingClient):
        async def close(self) -> None:
            raise RuntimeError("boom")

    session = SessionLog(tmp_path, meta={})
    agent = Agent(BrokenClient(), build_tool_registry([]), session=session)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_release_agent(agent))
    assert session.closed
