"""Tests for webhook listeners (webhooks.py).

The server tests drive a real ``ThreadingHTTPServer`` on an ephemeral port with
a capturing fake LLM client, so routing, bearer auth, body parsing, and the
agent round-trip are exercised end to end without any network.
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import socket
import textwrap
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from light_harness.agent import Agent
from light_harness.config import (
    AgentConfig,
    Config,
    Interaction,
    WebhookConfig,
    persona_sessions_dir,
)
from light_harness.llm import ChatMessage, LLMResponse
from light_harness.sessions import SessionLog
from light_harness.tools import build_tool_registry
from light_harness.webhooks import (
    WebhookHandler,
    _bearer_matches,
    create_webhook_server,
    parse_webhook_body,
    tool_availability_warning,
    unauthenticated_warning,
)

from ._helpers import join_thread


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
    join_thread(thread)


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


def _output_log_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, output_log: bool
) -> tuple[str, CapturingClient, ThreadingHTTPServer, threading.Thread]:
    """A started webhook server with ``webhook_server.output_log`` on or off."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    _write_cfg(
        tmp_path,
        f"""
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
        webhook_server:
          output_log: {str(output_log).lower()}
        """,
    )
    config = Config.load()
    client = CapturingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        persona = agent_cfg.name
        session = SessionLog(
            persona_sessions_dir(persona),
            meta={"persona": persona, "agent": persona, "model": "m", "source": "webhook"},
        )
        return Agent(
            client, build_tool_registry(agent_cfg.tools), system_prompt="sp", session=session
        )

    server = create_webhook_server(config, "127.0.0.1", 0, agent_factory=factory)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", client, server, thread


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
    # Non-ASCII secrets compare in bytes: the header text is what iso-8859-1
    # decoding produced from the wire, so the secret matches its own UTF-8
    # bytes (what a curl-style client sends) and no other encoding of itself.
    wire = "café".encode().decode("iso-8859-1")
    assert _bearer_matches("café", f"Bearer {wire}") is True
    assert _bearer_matches("café", "Bearer café") is False  # latin-1 bytes on the wire
    assert _bearer_matches("café", "Bearer caf\u00c3\u00a8") is False
    assert _bearer_matches("s3cret", f"Bearer {wire}") is False


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


def test_webhook_non_ascii_secret_authenticates_over_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1: a secret with non-ASCII must stay usable. The client sends its UTF-8
    bytes; the handler compares wire bytes against the secret's UTF-8 form
    instead of crashing on the iso-8859-1-decoded header."""
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
          - path: /hooks/uni
            agent: assistant
            secret: café
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

    def _post_raw(authorization: bytes) -> tuple[int, dict[str, object]]:
        conn = http.client.HTTPConnection(f"127.0.0.1:{port}", timeout=5)
        try:
            conn.putrequest("POST", "/hooks/uni")
            conn.putheader("Authorization", authorization)
            conn.putheader("Content-Length", "0")
            conn.endheaders()
            response = conn.getresponse()
            payload: dict[str, object] = json.loads(response.read().decode("utf-8"))
            return response.status, payload
        finally:
            conn.close()

    try:
        status, data = _post_raw("Bearer café".encode())
        assert status == 200
        assert data == {"agent": "assistant", "message": "OK"}
        _wait_for_prompts(client)
        assert client.prompts == [""]

        status, data = _post_raw("Bearer cafè".encode())
        assert status == 401
        assert data["message"] == "unauthorized"

        # The same secret sent as latin-1 bytes is not the secret's UTF-8 form.
        status, _ = _post_raw("Bearer café".encode("iso-8859-1"))
        assert status == 401
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


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
        join_thread(thread)


def test_webhook_get_is_405(webhook_server) -> None:
    base, _ = webhook_server
    status, _ = _get(base + "/hooks/open")
    assert status == 405


def _real_factory_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Path | None):
    """A real ``Config`` (personas, workspace, provider) for ``default_agent_factory``."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FACTORY_KEY", "sk-factory")
    _write_cfg(
        tmp_path / "home",
        "providers:\n"
        "  local:\n"
        "    base_url: http://llm.test/v1\n"
        "    api_key: $FACTORY_KEY\n"
        "    models: [m]\n"
        "personas:\n"
        "  p1:\n"
        "    SOUL.md: You are persona one.\n"
        "agents:\n"
        "  - name: assistant\n"
        "    provider: local\n"
        "    model: m\n"
        "    persona: p1\n"
        "    tools: [echo, read_file]\n"
        + (f"    workspace: {workspace}\n" if workspace else "")
        + "webhooks:\n"
        "  - path: /hooks/a\n"
        "    agent: assistant\n"
        "  - path: /hooks/b\n"
        "    agent: assistant\n"
        "    system_prompt: You are a deploy bot.\n",
    )
    config = Config.load()
    agent_cfg = config.get_agent("assistant")
    assert agent_cfg is not None
    return config, agent_cfg, {w.path: w for w in config.webhooks}


def test_default_factory_uses_the_personas_prompt_unless_the_webhook_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_harness.webhooks import _release_agent, default_agent_factory

    config, agent_cfg, hooks = _real_factory_setup(tmp_path, monkeypatch, None)

    plain = default_agent_factory(config, agent_cfg, hooks["/hooks/a"])
    overridden = default_agent_factory(config, agent_cfg, hooks["/hooks/b"])
    try:
        assert plain.system_prompt == config.system_prompt_for(agent_cfg)
        assert plain.system_prompt == "You are persona one."
        assert overridden.system_prompt == "You are a deploy bot."
    finally:
        asyncio.run(_release_agent(plain))
        asyncio.run(_release_agent(overridden))


def test_default_factory_creates_the_persona_and_a_webhook_session_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_harness.webhooks import _release_agent, default_agent_factory

    config, agent_cfg, hooks = _real_factory_setup(tmp_path, monkeypatch, None)

    agent = default_agent_factory(config, agent_cfg, hooks["/hooks/a"])
    try:
        assert agent.session is not None
        sessions = tmp_path / "home" / "personas" / "p1" / "sessions"
        assert agent.session.path.parent == sessions
        header = json.loads(agent.session.path.read_text(encoding="utf-8").splitlines()[0])
        assert header == {
            "type": "session",
            "persona": "p1",
            "agent": "assistant",
            "model": "m",
            "source": "webhook",
        }
        assert (tmp_path / "home" / "personas" / "p1" / "SOUL.md").is_file()
    finally:
        asyncio.run(_release_agent(agent))


def test_default_factory_wires_the_workspace_into_the_file_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from light_harness.webhooks import _release_agent, default_agent_factory

    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("from the workspace")
    config, agent_cfg, hooks = _real_factory_setup(tmp_path, monkeypatch, workspace)

    agent = default_agent_factory(config, agent_cfg, hooks["/hooks/a"])
    try:
        assert agent.tools.names() == ["echo", "read_file"]
        assert asyncio.run(agent.tools.run("read_file", {"path": "hello.txt"})) == (
            "from the workspace"
        )
        assert asyncio.run(agent.tools.run("read_file", {"path": "../../etc/passwd"})).startswith(
            "Error:"
        )
    finally:
        asyncio.run(_release_agent(agent))


def test_default_factory_closes_the_session_when_the_agent_cannot_be_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import light_harness.webhooks as webhooks_module

    config, agent_cfg, hooks = _real_factory_setup(tmp_path, monkeypatch, None)
    sessions: list = []
    real_session = webhooks_module.SessionLog

    def recording_session(*args, **kwargs):
        sessions.append(real_session(*args, **kwargs))
        return sessions[-1]

    def broken_agent(*args, **kwargs):
        raise RuntimeError("cannot build")

    monkeypatch.setattr(webhooks_module, "SessionLog", recording_session)
    monkeypatch.setattr(webhooks_module, "Agent", broken_agent)

    with pytest.raises(RuntimeError, match="cannot build"):
        webhooks_module.default_agent_factory(config, agent_cfg, hooks["/hooks/a"])

    assert len(sessions) == 1 and sessions[0].closed


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
    lines = [f"POST {path} HTTP/1.1", "Host: 127.0.0.1", "Transfer-Encoding: chunked"]
    lines += [f"{name}: {value}" for name, value in (extra or {}).items()]
    request = "\r\n".join(lines) + "\r\n\r\n"
    for chunk in chunks:
        request += f"{len(chunk):x}\r\n{chunk}\r\n"
    return (request + "0\r\n\r\n").encode()


def test_webhooks_reject_a_foreign_host_header(webhook_server) -> None:
    """S3: the webhook server had no Host check at all, so a page whose hostname has
    been made to resolve to 127.0.0.1 (DNS rebinding) could trigger every
    unauthenticated webhook from a browser."""
    url, client = webhook_server
    for method in ("POST", "GET", "OPTIONS"):
        request = f"{method} /hooks/open HTTP/1.1\r\nHost: evil.example\r\n\r\n".encode()
        status, headers = _raw_request(url, request)
        assert status == 403, method
        assert headers["content-type"] == "application/json"
    assert client.prompts == []
    # The same webhook, addressed the way its sender really addresses it, still runs.
    assert _post(url + "/hooks/open", b"hello")[0] == 200
    _wait_for_prompts(client)
    assert client.prompts == ["hello"]


def test_webhooks_answer_only_for_hosts_the_operator_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A webhook behind a reverse proxy is reached under a name the bind host does not
    cover; naming it admits that name, and every other hostname stays refused."""
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
        """,
    )
    config = Config.load()
    client = CapturingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        return Agent(client, build_tool_registry(agent_cfg.tools), system_prompt="sp")

    server = create_webhook_server(
        config, "127.0.0.1", 0, agent_factory=factory, allowed_hosts=["hooks.internal"]
    )
    url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for host, expected in (
            ("hooks.internal", 200),
            ("HOOKS.INTERNAL:8787", 200),
            ("127.0.0.1", 200),
            ("evil.example", 403),
        ):
            request = (
                f"POST /hooks/open HTTP/1.1\r\nHost: {host}\r\nContent-Length: 5\r\n\r\nhello"
            ).encode()
            assert _raw_request(url, request)[0] == expected, host
        _wait_for_prompts(client, 3)
        assert client.prompts == ["hello"] * 3
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


def test_webhook_rejects_oversized_body_without_reading_it(webhook_server) -> None:
    url, client = webhook_server
    status, headers = _raw_request(
        url, b"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 10485760\r\n\r\n"
    )
    assert status == 413
    # The oversized body is left on the socket, so the connection cannot be reused.
    assert headers["connection"] == "close"
    assert client.prompts == []


@pytest.mark.parametrize("value", ["abc", "-5"])
def test_webhook_rejects_malformed_content_length(webhook_server, value: str) -> None:
    url, client = webhook_server
    request = f"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {value}\r\n\r\n"
    status, headers = _raw_request(url, request.encode())
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
        b"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
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
        "POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
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
        "POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        f"{size_field}\r\nhello\r\n"
    ).encode()
    status, headers = _raw_request(url, request)
    assert status == 400
    assert headers["connection"] == "close"
    assert client.prompts == []


def test_transfer_encoding_other_than_chunked_is_refused(webhook_server) -> None:
    url, client = webhook_server
    request = (
        b"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: gzip\r\n"
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
    follow_up = b"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\nhi"
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
            b"POST /hooks/nope HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 5\r\n\r\nhello"
            b"POST /hooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\nhi"
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
    gate_timeouts: list[bool] = []

    class BlockingClient(CapturingClient):
        async def chat(self, messages, tools=None) -> LLMResponse:
            opened = await asyncio.get_running_loop().run_in_executor(None, gate.wait, 5)
            if not opened:  # Event.wait reports a timeout by returning False
                gate_timeouts.append(True)
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
        join_thread(thread)
    assert gate_timeouts == [], "a run waited the full gate timeout: the test never released it"


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


def _serve_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory, max_runs: int):
    """A webhook server for ``/hooks/open`` using ``factory`` (caller stops it)."""
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
    server = create_webhook_server(
        Config.load(), "127.0.0.1", 0, agent_factory=factory, max_concurrent_runs=max_runs
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}/hooks/open"


def _post_until_accepted(url: str, body: bytes, timeout: float = 5.0) -> None:
    """Retry while the server answers 429; fails if no slot frees up in time."""
    deadline = time.monotonic() + timeout
    while (status := _post(url, body)[0]) == 429:
        assert time.monotonic() < deadline, "the run slot was never released"
        time.sleep(0.01)
    assert status == 200


def test_webhook_admits_as_many_runs_as_max_concurrent_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    started = threading.Semaphore(0)
    gate_timeouts: list[bool] = []

    class BlockingClient(CapturingClient):
        async def chat(self, messages, tools=None) -> LLMResponse:
            started.release()
            if not await asyncio.get_running_loop().run_in_executor(None, gate.wait, 5):
                gate_timeouts.append(True)
            return await super().chat(messages, tools)

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        return Agent(BlockingClient(), build_tool_registry(agent_cfg.tools))

    server, thread, url = _serve_with(tmp_path, monkeypatch, factory, max_runs=2)
    try:
        assert _post(url, b"one")[0] == 200
        assert _post(url, b"two")[0] == 200
        assert started.acquire(timeout=5) and started.acquire(timeout=5)  # both really run
        assert _post(url, b"three")[0] == 429  # two slots, both held
        gate.set()
        _post_until_accepted(url, b"four")
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        join_thread(thread)
    assert gate_timeouts == []


def test_a_failing_run_releases_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []

    class FailingClient(CapturingClient):
        async def chat(self, messages, tools=None) -> LLMResponse:
            attempts.append("run")
            raise RuntimeError("model is down")

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        return Agent(FailingClient(), build_tool_registry(agent_cfg.tools))

    server, thread, url = _serve_with(tmp_path, monkeypatch, factory, max_runs=1)
    try:
        assert _post(url, b"one")[0] == 200  # accepted; the run then fails in the background
        _post_until_accepted(url, b"two")  # only possible once the failed run freed the slot
        deadline = time.monotonic() + 5
        while len(attempts) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


def test_a_factory_that_raises_releases_its_slot_and_answers_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    client = CapturingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig, webhook: WebhookConfig) -> Agent:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cannot build agent")
        return Agent(client, build_tool_registry(agent_cfg.tools))

    server, thread, url = _serve_with(tmp_path, monkeypatch, factory, max_runs=1)
    try:
        status, body = _post(url, b"one")
        assert status == 500
        assert "cannot build agent" in body["message"]
        assert _post(url, b"two")[0] == 200  # the slot was returned, not leaked
        _wait_for_prompts(client)
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


def test_output_log_prints_a_line_per_accepted_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, client, server, thread = _output_log_server(tmp_path, monkeypatch, output_log=True)
    try:
        assert _post(base + "/hooks/open", b'{"a": 1}')[0] == 200
        _wait_for_prompts(client)
        assert _post(base + "/hooks/missing", b"{}")[0] == 404
        assert _post(base + "/hooks/secret", b"{}")[0] == 401
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)

    out = capsys.readouterr().out
    logged = [line for line in out.splitlines() if "webhook POST" in line]
    assert len(logged) == 1  # rejected calls (404, 401) log nothing
    assert "webhook POST /hooks/open -> agent 'assistant'" in logged[0]
    assert "(session: personas/assistant/sessions/" in logged[0]
    assert logged[0].endswith(".jsonl)")


def test_output_log_off_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, client, server, thread = _output_log_server(tmp_path, monkeypatch, output_log=False)
    try:
        assert _post(base + "/hooks/open", b"{}")[0] == 200
        _wait_for_prompts(client)
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)
    assert "webhook POST" not in capsys.readouterr().out
