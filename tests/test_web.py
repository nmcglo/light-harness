"""Tests for the config web UI (web.py).

These drive a real ``ThreadingHTTPServer`` (on an ephemeral port) over HTTP,
so the request handling, file I/O, and round-trip of values are exercised
end to end.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from light_harness.web import create_server


@pytest.fixture()
def web_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Start a real config web server on an ephemeral port; yield its base URL."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    (tmp_path / "config.yaml").unlink(missing_ok=True)
    server = create_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _token(url: str) -> str:
    """The API token a real browser would read from the served page."""
    base = url.split("/api/", 1)[0]
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        match = re.search(r'name="agent-token" content="([^"]+)"', r.read().decode("utf-8"))
    assert match
    return match.group(1)


def _request(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    token: bool = True,
) -> tuple[int, dict[str, str], str]:
    """One request -> (status, response headers, body). Sends the page token on
    ``/api/`` URLs unless ``token`` is False."""
    headers = dict(headers or {})
    if token and "/api/" in url:
        headers["X-Agent-Token"] = _token(url)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8")


def _get(url: str) -> tuple[int, str]:
    status, _, body = _request(url)
    return status, body


def _get_json(url: str) -> tuple[int, dict]:
    status, body = _get(url)
    return status, json.loads(body)


def _post(url: str, payload: object) -> tuple[int, dict]:
    data = (
        payload.encode("utf-8")
        if isinstance(payload, str)
        else json.dumps(payload).encode("utf-8")
    )
    status, _, body = _request(url, "POST", data, {"Content-Type": "application/json"})
    return status, json.loads(body)


def _post_raw(url: str, body: bytes) -> tuple[int, dict]:
    """POST a byte body exactly as given (so an empty body stays empty)."""
    status, _, text = _request(url, "POST", body, {"Content-Type": "application/json"})
    return status, json.loads(text)


def _post_chunked(
    base: str, payload: object, path: str = "/api/config"
) -> tuple[int, dict[str, str]]:
    """POST a JSON body framed as chunked transfer encoding (urllib can only send
    Content-Length); returns the status and the response headers."""
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    try:
        conn.request(
            "POST",
            path,
            f"{len(body):x}\r\n{body}\r\n0\r\n\r\n",
            {"X-Agent-Token": _token(base), "Transfer-Encoding": "chunked"},
        )
        response = conn.getresponse()
        return response.status, dict(response.headers)
    finally:
        conn.close()


def test_api_requires_token(web_base: str) -> None:
    for method, data in (("GET", None), ("POST", b"{}")):
        status, _, body = _request(web_base + "/api/config", method, data, token=False)
        assert status == 401, method
        assert "X-Agent-Token" in body
    status, _, _ = _request(
        web_base + "/api/config", headers={"X-Agent-Token": "wrong"}, token=False
    )
    assert status == 401


def test_non_ascii_token_is_rejected_without_crashing(web_base: str) -> None:
    """A1: a header value that decodes to non-ASCII must get a 401, not a
    TypeError traceback and a dropped connection. The server decodes headers as
    iso-8859-1, so the compared value is whatever bytes hit the wire."""
    conn = http.client.HTTPConnection(web_base.removeprefix("http://"), timeout=5)
    try:
        for token in ("café".encode(), "caf©".encode("iso-8859-1")):
            conn.putrequest("GET", "/api/config")
            conn.putheader("X-Agent-Token", token)
            conn.endheaders()
            response = conn.getresponse()
            assert response.status == 401, token
            assert "X-Agent-Token" in response.read().decode("utf-8")
    finally:
        conn.close()


def test_cors_defaults_to_wildcard_on_api_but_not_on_page(web_base: str) -> None:
    _, headers, _ = _request(web_base + "/api/tools")
    assert headers["Access-Control-Allow-Origin"] == "*"
    # 401s are readable cross-origin too, so a browser client sees the reason.
    _, headers, _ = _request(web_base + "/api/tools", token=False)
    assert headers["Access-Control-Allow-Origin"] == "*"
    # The page embeds the token, so it must not be readable cross-origin.
    _, headers, _ = _request(web_base + "/")
    assert "Access-Control-Allow-Origin" not in headers
    assert headers["Cache-Control"] == "no-store"


def test_cors_preflight(web_base: str) -> None:
    status, headers, _ = _request(
        web_base + "/api/config",
        "OPTIONS",
        headers={"Origin": "http://other.example", "Access-Control-Request-Method": "POST"},
        token=False,
    )
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "X-Agent-Token" in headers["Access-Control-Allow-Headers"]
    assert "POST" in headers["Access-Control-Allow-Methods"]


def test_cors_origin_configurable_and_disableable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    for configured, expected in (("http://ui.example", "http://ui.example"), ("", None)):
        server = create_server("127.0.0.1", 0, cors_origin=configured)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/api/tools"
            _, headers, _ = _request(url)
            assert headers.get("Access-Control-Allow-Origin") == expected
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_rejects_unexpected_host_header(web_base: str) -> None:
    # DNS rebinding: attacker's name resolves to 127.0.0.1 but keeps its own Host.
    for path in ("/", "/api/tools"):
        status, _, _ = _request(web_base + path, headers={"Host": "evil.example"})
        assert status == 403, path
    status, _, _ = _request(web_base + "/api/tools", headers={"Host": "localhost:8765"})
    assert status == 200


def test_serves_html_page(web_base: str) -> None:
    status, body = _get(web_base + "/")
    assert status == 200
    assert "light-harness" in body
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


_SEEDED_CONFIG = """\
# a comment only a refused save keeps
providers:
  local:
    base_url: http://x/v1
    models:
      - qwen
agents:
  - name: a
    provider: local
    model: qwen
personas:
  p1:
    SOUL.md: You are kind.
    EXAMPLES.md: keep me
interaction:
  chat:
    agent: a
webhooks:
  - path: /hooks/x
"""


def _seed(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SEEDED_CONFIG, encoding="utf-8")
    return cfg


def test_post_empty_body_changes_nothing(web_base: str, tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    status, data = _post_raw(web_base + "/api/config", b"")
    assert status == 400
    assert data["ok"] is False
    assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG


def test_post_empty_object_changes_nothing(web_base: str, tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    status, data = _post(web_base + "/api/config", {})
    assert status == 400
    assert data["ok"] is False
    assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG


def test_post_keeps_sections_the_body_does_not_name(web_base: str, tmp_path: Path) -> None:
    _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config",
        {"agents": [{"name": "a", "provider": "local", "model": "qwen", "tools": []}]},
    )
    assert status == 200, data
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["providers"]["local"]["models"] == ["qwen"]
    assert parsed["personas"]["p1"]["SOUL.md"] == "You are kind."
    assert parsed["interaction"]["chat"]["agent"] == "a"
    assert parsed["webhooks"][0]["path"] == "/hooks/x"
    assert parsed["agents"][0]["tools"] == []


def test_post_null_section_is_unchanged_not_cleared(web_base: str, tmp_path: Path) -> None:
    _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config",
        {"providers": None, "agents": [{"name": "b", "model": "m"}]},
    )
    assert status == 200, data
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["providers"]["local"]["base_url"] == "http://x/v1"
    assert [a["name"] for a in parsed["agents"]] == ["b"]


def test_post_named_section_replaces_it(web_base: str, tmp_path: Path) -> None:
    """A section the body sends replaces the old one; an empty list clears it."""
    _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config",
        {
            "providers": [],
            "agents": [{"name": "a", "provider": "local", "model": "qwen", "tools": None}],
            "personas": [{"name": "p1", "SOUL.md": "You are kind."}],
            "interaction": {"chat": {"agent": "a"}},
        },
    )
    assert status == 200, data
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "providers" not in parsed
    assert parsed["agents"][0]["model"] == "qwen"
    assert parsed["webhooks"][0]["path"] == "/hooks/x"


def test_post_rejects_the_files_own_section_shape(web_base: str, tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    for payload in (
        {"providers": {"local": {"base_url": "http://x/v1"}}},
        {"personas": {"p1": {"SOUL.md": "You are kind."}}},
        {"interaction": "a"},
    ):
        status, data = _post(web_base + "/api/config", payload)
        assert status == 400, payload
        assert data["ok"] is False
        assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG


def test_post_rejects_entries_without_a_name(web_base: str, tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    for payload in (
        {"providers": [{"base_url": "http://x/v1"}]},
        {"personas": [{"SOUL.md": "You are kind."}]},
    ):
        status, data = _post(web_base + "/api/config", payload)
        assert status == 400, payload
        assert "name" in data["error"]
        assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG


def test_post_rejects_non_string_persona_content(web_base: str, tmp_path: Path) -> None:
    cfg = _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config", {"personas": [{"name": "p1", "SOUL.md": 12}]}
    )
    assert status == 400
    assert "SOUL.md" in data["error"]
    assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG


def test_post_keeps_persona_files_the_form_cannot_edit(web_base: str, tmp_path: Path) -> None:
    _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config",
        {"personas": [{"name": "p1", "SOUL.md": "You are sharper."}]},
    )
    assert status == 200, data
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["personas"]["p1"]["SOUL.md"] == "You are sharper."
    assert parsed["personas"]["p1"]["EXAMPLES.md"] == "keep me"


def test_chunked_post_body_writes_the_file(web_base: str, tmp_path: Path) -> None:
    """A chunked body is decoded, so the form's JSON reaches the parser instead of
    reading as empty."""
    status, _ = _post_chunked(web_base, {"agents": [{"name": "a", "model": "m"}]})
    assert status == 200
    data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert data == {"agents": [{"name": "a", "model": "m"}]}


def test_post_with_an_unread_body_closes_the_connection(web_base: str) -> None:
    """The 404 is answered before the body is read, so the connection must close:
    its bytes would otherwise be parsed as the next request."""
    conn = http.client.HTTPConnection(web_base.removeprefix("http://"), timeout=5)
    try:
        conn.request("POST", "/api/other", "hello", {"X-Agent-Token": _token(web_base)})
        response = conn.getresponse()
        assert response.status == 404
        assert response.getheader("Connection") == "close"
    finally:
        conn.close()


