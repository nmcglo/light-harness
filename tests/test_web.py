"""Tests for the config web UI (web.py).

These drive a real ``ThreadingHTTPServer`` (on an ephemeral port) over HTTP,
so the request handling, file I/O, and round-trip of values are exercised
end to end.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from agent_responder.web import create_server


@pytest.fixture()
def web_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Start a real config web server on an ephemeral port; yield its base URL."""
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    (tmp_path / "config.yaml").unlink(missing_ok=True)
    server = create_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _get_json(url: str) -> tuple[int, dict]:
    status, body = _get(url)
    return status, json.loads(body)


def _post(url: str, payload: object) -> tuple[int, dict]:
    data = (
        payload.encode("utf-8")
        if isinstance(payload, str)
        else json.dumps(payload).encode("utf-8")
    )
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_serves_html_page(web_base: str) -> None:
    status, body = _get(web_base + "/")
    assert status == 200
    assert "agent-responder" in body
    assert "provider-list" in body
    assert "agent-provider" in body


def test_get_tools(web_base: str) -> None:
    status, data = _get_json(web_base + "/api/tools")
    assert status == 200
    assert data["tools"] == ["echo", "current_time", "add", "read_file", "write_file", "edit_file"]


def test_get_config_absent(web_base: str) -> None:
    status, data = _get_json(web_base + "/api/config")
    assert status == 200
    assert data["exists"] is False
    assert data["text"] is None
    assert data["data"] == {}


def test_get_config_present(web_base: str, tmp_path: Path) -> None:
    text = (
        "providers:\n  local:\n    base_url: http://x/v1\n    models:\n      - qwen\n"
        "agents:\n  - name: a\n    provider: local\n    model: qwen\n"
    )
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
    status, data = _get_json(web_base + "/api/config")
    assert status == 200
    assert data["exists"] is True
    assert data["data"]["providers"]["local"]["models"] == ["qwen"]
    assert data["data"]["agents"][0]["model"] == "qwen"


def test_post_writes_file(web_base: str, tmp_path: Path) -> None:
    payload = {
        "providers": [
            {
                "name": "local",
                "base_url": "http://x/v1",
                "api_key": "$MY_KEY",
                "timeout": "30",
                "models": ["qwen", "gpt"],
            }
        ],
        "agents": [{"name": "a", "provider": "local", "model": "qwen", "tools": None}],
    }
    status, data = _post(web_base + "/api/config", payload)
    assert status == 200
    assert data["ok"] is True
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["providers"]["local"]["api_key"] == "$MY_KEY"
    assert parsed["providers"]["local"]["models"] == ["qwen", "gpt"]
    assert parsed["providers"]["local"]["timeout"] == 30
    assert "tools" not in parsed["agents"][0]


def test_post_tools_empty_list(web_base: str, tmp_path: Path) -> None:
    payload = {"agents": [{"name": "a", "provider": "local", "model": "m", "tools": []}]}
    status, _ = _post(web_base + "/api/config", payload)
    assert status == 200
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["agents"][0]["tools"] == []


def test_post_preserves_env_refs(web_base: str, tmp_path: Path) -> None:
    payload = {
        "providers": [{"name": "local", "api_key": "${MY_KEY}", "models": ["$M"]}],
        "agents": [{"name": "a", "provider": "local", "model": "m", "tools": None}],
    }
    status, _ = _post(web_base + "/api/config", payload)
    assert status == 200
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "${MY_KEY}" in written
    assert "$M" in written


def test_post_preserves_webhooks(web_base: str, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "agents:\n  - name: a\n    model: m\n"
        "webhooks:\n  - path: /hooks/x\n    agent: a\n",
        encoding="utf-8",
    )
    payload = {"agents": [{"name": "a", "provider": "local", "model": "m", "tools": None}]}
    status, _ = _post(web_base + "/api/config", payload)
    assert status == 200
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "webhooks:" in written
    assert "/hooks/x" in written


def test_post_rejects_invalid_json(web_base: str, tmp_path: Path) -> None:
    status, data = _post(web_base + "/api/config", "{not json")
    assert status == 400
    assert data["ok"] is False
    assert not (tmp_path / "config.yaml").exists()


def test_post_rejects_non_object(web_base: str, tmp_path: Path) -> None:
    status, data = _post(web_base + "/api/config", json.dumps([1, 2]))
    assert status == 400
    assert data["ok"] is False
    assert not (tmp_path / "config.yaml").exists()


def test_external_file_change_reflected(web_base: str, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("agents:\n  - name: a\n    model: a\n", encoding="utf-8")
    _, first = _get_json(web_base + "/api/config")
    cfg.write_text("agents:\n  - name: a\n    model: b\n", encoding="utf-8")
    _, second = _get_json(web_base + "/api/config")
    assert first["text"] == "agents:\n  - name: a\n    model: a\n"
    assert second["text"] == "agents:\n  - name: a\n    model: b\n"


def test_post_agents_personas_interaction(web_base: str, tmp_path: Path) -> None:
    payload = {
        "providers": [{"name": "local", "base_url": "http://x/v1", "models": ["qwen"]}],
        "agents": [
            {"name": "a", "provider": "local", "model": "qwen", "persona": "p1"},
            {"name": "b", "provider": "local", "model": "gpt", "tools": ["echo"]},
        ],
        "personas": [{"name": "p1", "SOUL.md": "You are kind."}],
        "interaction": {"chat": {"agent": "b"}},
    }
    status, data = _post(web_base + "/api/config", payload)
    assert status == 200
    assert data["ok"] is True
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert [a["name"] for a in parsed["agents"]] == ["a", "b"]
    assert parsed["agents"][0]["persona"] == "p1"
    assert parsed["agents"][1]["tools"] == ["echo"]
    assert parsed["personas"]["p1"]["SOUL.md"] == "You are kind."
    assert parsed["interaction"]["chat"]["agent"] == "b"


def test_post_agent_requires_name(web_base: str, tmp_path: Path) -> None:
    payload = {"agents": [{"provider": "local", "model": "m"}]}
    status, data = _post(web_base + "/api/config", payload)
    assert status == 400
    assert data["ok"] is False
    assert not (tmp_path / "config.yaml").exists()
