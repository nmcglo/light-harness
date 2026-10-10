"""Tests for webhook listeners (webhooks.py).

The server tests drive a real ``ThreadingHTTPServer`` on an ephemeral port with
a capturing fake LLM client, so routing, bearer auth, body parsing, and the
agent round-trip are exercised end to end without any network.
"""

from __future__ import annotations

import json
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agent_responder.agent import Agent
from agent_responder.config import AgentConfig, Config
from agent_responder.llm import ChatMessage, LLMResponse
from agent_responder.tools import build_tool_registry
from agent_responder.webhooks import (
    _bearer_matches,
    create_webhook_server,
    parse_webhook_body,
)

_ALL_VARS = (
    "AGENT_MODEL",
    "AGENT_PROVIDER",
    "AGENT_BASE_URL",
    "AGENT_API_KEY",
    "AGENT_TIMEOUT",
    "OPENAI_API_KEY",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_cfg(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")


class CapturingClient:
    """A fake LLM client: records each user prompt and returns a fixed answer."""

    def __init__(self, answer: str = "handled") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def chat(self, messages, tools=None) -> LLMResponse:
        for message in messages:
            if message.role == "user":
                self.prompts.append(message.content)
        return LLMResponse(
            message=ChatMessage(role="assistant", content=self.answer), tool_calls=[]
        )

    async def close(self) -> None:
        return None


@pytest.fixture()
def webhook_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A webhook server on an ephemeral port backed by a capturing fake client."""
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
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
        """,
    )
    config = Config.load()
    client = CapturingClient()

    def factory(cfg: Config, agent_cfg: AgentConfig) -> Agent:
        return Agent(client, build_tool_registry(agent_cfg.tools), system_prompt="sp")

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


def test_config_load_parses_webhooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
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
        """,
    )
    config = Config.load()
    assert [w.path for w in config.webhooks] == ["/hooks/deploy", "/hooks/plain"]
    deploy = config.webhook_by_path("/hooks/deploy")
    assert deploy is not None and deploy.secret == "tok"
    plain = config.webhook_by_path("/hooks/plain")
    assert plain is not None and plain.secret is None
    assert config.webhook_by_path("/nope") is None
    assert config.get_agent("assistant") is not None


def test_parse_webhooks_drops_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    _write_cfg(
        tmp_path,
        """
        webhooks:
          - path: /hooks/ok
            agent: assistant
          - path: not-a-path
            agent: assistant
          - agent: assistant
          - path: /hooks/ok
            agent: assistant
        """,
    )
    config = Config.load()
    assert [w.path for w in config.webhooks] == ["/hooks/ok"]


def test_endpoint_for_scopes_to_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
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
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
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
