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

from light_harness.config import Config
from light_harness.web import create_server

from ._helpers import join_thread


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
    join_thread(thread)


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
            join_thread(thread)


def test_rejects_unexpected_host_header(web_base: str) -> None:
    # DNS rebinding: attacker's name resolves to 127.0.0.1 but keeps its own Host.
    for path in ("/", "/api/tools"):
        status, _, _ = _request(web_base + path, headers={"Host": "evil.example"})
        assert status == 403, path
    status, _, _ = _request(web_base + "/api/tools", headers={"Host": "localhost:8765"})
    assert status == 200


def test_cors_origin_must_be_a_bare_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S3: the allow-origin is echoed into a response header verbatim, so a value
    carrying CRLF appends a header of the caller's choosing --
    ``Access-Control-Allow-Credentials: true``, say -- to every ``/api/*`` answer,
    which carries the config's API keys and webhook secrets."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    injected = "http://ui.example\r\nAccess-Control-Allow-Credentials: true"
    for bad in (
        injected,
        "http://ui.example\nX-Injected: 1",
        "http://ui.example/path",
        "http://ui.example?next=1",
        "http://ui.example#frag",
        "http://user:pass@ui.example",
        "http://ui.example ",
        "http://ui.example:port",
        "ui.example",
        "null",
        "javascript:alert(1)",
    ):
        with pytest.raises(ValueError, match="Invalid CORS allow-origin"):
            create_server("127.0.0.1", 0, cors_origin=bad)
    for good in (
        "*",
        "http://ui.example",
        "https://ui.example:8443",
        "http://127.0.0.1:3000",
        "https://[::1]:8080",
    ):
        server = create_server("127.0.0.1", 0, cors_origin=good)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/api/tools"
            _, headers, _ = _request(url)
            assert headers.get("Access-Control-Allow-Origin") == good
        finally:
            server.shutdown()
            server.server_close()
            join_thread(thread)


def test_api_responses_are_not_cacheable(web_base: str) -> None:
    """S3: ``/api/config`` returns the config file verbatim -- API keys and webhook
    secrets -- but only the page was sent with ``Cache-Control: no-store``."""
    cases = (
        ("GET", "/api/config", True, None),
        ("GET", "/api/tools", True, None),
        ("GET", "/api/tools", False, None),  # a 401 is a response too
        ("GET", "/nope", False, None),  # a 404
        ("POST", "/api/config", True, b"not json"),  # a 400
    )
    for method, path, token, data in cases:
        status, headers, _ = _request(web_base + path, method, data, token=token)
        assert status in (200, 400, 401, 404), (method, path, status)
        assert headers["Cache-Control"] == "no-store", (method, path, status)


def test_wildcard_bind_still_checks_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3: binding ``0.0.0.0`` used to switch the Host check off entirely -- the bind
    a browser can be talked into reaching under a hostname an attacker controls."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    server = create_server("0.0.0.0", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/", "/api/config", "/api/tools"):
            status, _, _ = _request(base + path, headers={"Host": "evil.example"})
            assert status == 403, path
        # The same server, addressed the way a LAN client actually addresses it.
        assert _get(base + "/")[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


def test_allowed_hosts_name_the_names_the_server_is_reached_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server reached under a name the bind host does not cover (mDNS, a reverse
    proxy) needs that named; every other hostname stays refused, and an IP address
    -- which no rebinding attack can put in a browser's ``Host`` -- needs no naming."""
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(tmp_path))
    server = create_server("127.0.0.1", 0, allowed_hosts=["ui.internal", "Box.Local"])
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for host in (
            "ui.internal",
            "ui.internal:8765",
            "UI.INTERNAL",
            "box.local",
            "localhost",
            "app.localhost",
            "192.168.1.5",
            "[::1]",
        ):
            status, _, _ = _request(base + "/api/tools", headers={"Host": host})
            assert status == 200, host
        for host in ("evil.example", "evil.example:8765", "ui.internal.evil.example"):
            status, _, _ = _request(base + "/api/tools", headers={"Host": host})
            assert status == 403, host
    finally:
        server.shutdown()
        server.server_close()
        join_thread(thread)


def test_serves_html_page(web_base: str) -> None:
    status, body = _get(web_base + "/")
    assert status == 200
    assert "light-harness" in body
    assert "provider-list" in body
    assert "agent-provider" in body


def test_get_tools(web_base: str) -> None:
    status, data = _get_json(web_base + "/api/tools")
    assert status == 200
    assert data["tools"] == ["current_time", "read_file", "write_file", "edit_file"]


def test_get_tools_lists_forgejo_tools_separately(web_base: str) -> None:
    _, data = _get_json(web_base + "/api/tools")
    assert data["forgejo_tools"] == [
        "forgejo_pr",
        "forgejo_pr_diff",
        "forgejo_pr_comment",
        "forgejo_pr_comment_edit",
        "forgejo_pr_review",
        "forgejo_pr_comment_delete",
        "forgejo_pr_review_delete",
        "forgejo_pr_create",
        "forgejo_pr_close",
        "forgejo_pr_merge",
        "forgejo_issue_create",
        "forgejo_issue_list",
        "forgejo_issue",
        "forgejo_issue_update",
        "forgejo_issue_comment",
        "forgejo_issue_comment_edit",
        "forgejo_issue_comment_delete",
        "forgejo_clone",
    ]


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
            {"name": "b", "provider": "local", "model": "gpt", "tools": ["current_time"]},
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
    assert parsed["agents"][1]["tools"] == ["current_time"]
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


def test_persona_edit_through_the_ui_changes_the_system_prompt(
    web_base: str, tmp_path: Path
) -> None:
    """The persona field edits ``personas/<name>/SOUL.md`` -- the file the agent
    reads -- not only the config value that seeds it."""
    status, data = _post(
        web_base + "/api/config",
        {
            "agents": [{"name": "a", "provider": "local", "model": "qwen", "persona": "p1"}],
            "personas": [{"name": "p1", "SOUL.md": "You are sharper."}],
        },
    )
    assert status == 200, data
    soul = tmp_path / "personas" / "p1" / "SOUL.md"
    assert soul.read_text(encoding="utf-8") == "You are sharper."

    cfg = Config.load()
    assert cfg.system_prompt_for(cfg.active_agent()) == "You are sharper."

    # The form now shows the prompt in force, so editing it a second time works too.
    status, shown = _get_json(web_base + "/api/config")
    assert status == 200
    assert shown["data"]["personas"]["p1"]["SOUL.md"] == "You are sharper."
    status, data = _post(
        web_base + "/api/config",
        {"personas": [{"name": "p1", "SOUL.md": "You are sharpest."}]},
    )
    assert status == 200, data
    cfg = Config.load()
    assert cfg.system_prompt_for(cfg.active_agent()) == "You are sharpest."


def test_get_config_shows_the_stored_persona_soul_not_the_seed(
    web_base: str, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("personas:\n  p1:\n    SOUL.md: seeded text\n", encoding="utf-8")
    soul = tmp_path / "personas" / "p1" / "SOUL.md"
    soul.parent.mkdir(parents=True)
    soul.write_text("edited on disk", encoding="utf-8")

    status, got = _get_json(web_base + "/api/config")
    assert status == 200
    assert got["data"]["personas"]["p1"]["SOUL.md"] == "edited on disk"
    # The config file itself is still reported verbatim.
    assert "seeded text" in got["text"]


def test_get_config_shows_the_seed_until_the_persona_exists(web_base: str, tmp_path: Path) -> None:
    """Reading the config through the API neither seeds nor creates a persona."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("personas:\n  p1:\n    SOUL.md: seeded text\n", encoding="utf-8")
    status, got = _get_json(web_base + "/api/config")
    assert status == 200
    assert got["data"]["personas"]["p1"]["SOUL.md"] == "seeded text"
    assert not (tmp_path / "personas").exists()


def test_clearing_the_persona_field_clears_the_stored_prompt(
    web_base: str, tmp_path: Path
) -> None:
    _seed(tmp_path)
    soul = tmp_path / "personas" / "p1" / "SOUL.md"
    soul.parent.mkdir(parents=True)
    soul.write_text("You are kind.", encoding="utf-8")

    status, data = _post(web_base + "/api/config", {"personas": [{"name": "p1", "SOUL.md": ""}]})
    assert status == 200, data
    assert soul.read_text(encoding="utf-8") == ""
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "SOUL.md" not in parsed["personas"]["p1"]
    assert parsed["personas"]["p1"]["EXAMPLES.md"] == "keep me"


def test_persona_entry_without_soul_leaves_the_stored_prompt(
    web_base: str, tmp_path: Path
) -> None:
    """An entry that does not send ``SOUL.md`` saves its other fields only."""
    _seed(tmp_path)
    soul = tmp_path / "personas" / "p1" / "SOUL.md"
    soul.parent.mkdir(parents=True)
    soul.write_text("prompt on disk", encoding="utf-8")

    status, data = _post(
        web_base + "/api/config",
        {"personas": [{"name": "p1", "EXAMPLES.md": "rewritten examples"}]},
    )
    assert status == 200, data
    assert soul.read_text(encoding="utf-8") == "prompt on disk"
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert parsed["personas"]["p1"]["EXAMPLES.md"] == "rewritten examples"


def test_post_rejects_unsafe_persona_name(web_base: str, tmp_path: Path) -> None:
    """A persona name becomes a directory name, so it is refused before anything
    is written."""
    cfg = _seed(tmp_path)
    status, data = _post(
        web_base + "/api/config", {"personas": [{"name": "../evil", "SOUL.md": "x"}]}
    )
    assert status == 400, data
    assert "persona name" in data["error"]
    assert cfg.read_text(encoding="utf-8") == _SEEDED_CONFIG
    assert not (tmp_path.parent / "evil").exists()
    assert not (tmp_path / "personas").exists()


def test_persona_soul_env_ref_survives_a_save(
    web_base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persona file holds the resolved prompt; the config keeps the reference."""
    monkeypatch.setenv("MY_PROMPT", "You are resolved.")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "agents:\n  - name: a\n    persona: p1\n"
        "personas:\n  p1:\n    SOUL.md: $MY_PROMPT\n",
        encoding="utf-8",
    )
    Config.load()  # first load seeds the persona file with the resolved text
    soul = tmp_path / "personas" / "p1" / "SOUL.md"
    assert soul.read_text(encoding="utf-8") == "You are resolved."

    status, got = _get_json(web_base + "/api/config")
    assert status == 200
    assert got["data"]["personas"]["p1"]["SOUL.md"] == "$MY_PROMPT"

    status, data = _post(
        web_base + "/api/config",
        {"personas": [{"name": "p1", "SOUL.md": "$MY_PROMPT be sharp."}]},
    )
    assert status == 200, data
    assert "$MY_PROMPT be sharp." in cfg.read_text(encoding="utf-8")
    assert soul.read_text(encoding="utf-8") == "You are resolved. be sharp."


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




def test_binary_config_file_gets_json_errors_on_get_and_post(
    web_base: str, tmp_path: Path
) -> None:
    """A config file that is not UTF-8 used to make GET raise inside the handler
    (no response at all); both verbs now answer with a JSON error."""
    cfg = tmp_path / "config.yaml"
    blob = b"\xff\xfe\x00binary"
    cfg.write_bytes(blob)

    status, data = _get_json(web_base + "/api/config")
    assert status == 500
    assert "Cannot read" in data["error"]

    status, data = _post(web_base + "/api/config", {"providers": []})
    assert status == 400
    assert data["ok"] is False
    assert cfg.read_bytes() == blob


def _raw_post(web_base: str, head_lines: list[str], tail: bytes = b"") -> tuple[int, dict, bytes]:
    """POST /api/config over a raw socket with exactly the given header lines (no
    automatic Content-Length), returning (status, lowercased headers, body)."""
    import socket

    host, port = web_base.removeprefix("http://").split(":")
    lines = [
        "POST /api/config HTTP/1.1",
        f"Host: {host}:{port}",
        f"X-Agent-Token: {_token(web_base)}",
        "Content-Type: application/json",
        *head_lines,
    ]
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + tail)
        stream = s.makefile("rb")
        status = int(stream.readline().split()[1])
        headers: dict[str, str] = {}
        while (line := stream.readline()) not in (b"\r\n", b""):
            name, _, value = line.decode("iso-8859-1").partition(":")
            headers[name.lower()] = value.strip()
        body = stream.read(int(headers.get("content-length", "0")))
    return status, headers, body


_SEED = "agents:\n  - name: keep\n    model: m\n"


@pytest.mark.parametrize(
    ("head_lines", "tail", "status", "needle"),
    [
        (["Content-Length: 10485760"], b"", 413, "exceeds"),
        (["Content-Length: abc"], b"", 400, "Content-Length"),
        (["Content-Length: -5"], b"", 400, "Content-Length"),
        (["Transfer-Encoding: gzip"], b"", 400, "Transfer-Encoding"),
        (["Transfer-Encoding: chunked"], b"zz\r\n", 400, "chunk size"),
        (["Transfer-Encoding: chunked"], b"200001\r\n", 413, "exceeds"),
    ],
)
def test_post_body_errors_are_json_closed_and_leave_the_config_alone(
    web_base: str,
    tmp_path: Path,
    head_lines: list[str],
    tail: bytes,
    status: int,
    needle: str,
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SEED, encoding="utf-8")

    got_status, headers, body = _raw_post(web_base, head_lines, tail)

    assert got_status == status
    assert headers["connection"] == "close"  # the unread bytes must not become a request
    payload = json.loads(body)
    assert payload["ok"] is False
    assert needle in payload["error"]
    assert cfg.read_text(encoding="utf-8") == _SEED


def _js_runtime() -> str | None:
    import shutil

    return shutil.which("node")


@pytest.mark.skipif(_js_runtime() is None, reason="needs node to run the page's JavaScript")
def test_the_pages_esc_function_neutralizes_markup() -> None:
    """Config values reach the DOM through ``innerHTML``; ``esc`` is what stops a
    stored value such as a persona prompt from running script with the API token."""
    import subprocess

    from light_harness.web import _PAGE_JS

    start = _PAGE_JS.index("function esc(")
    end = _PAGE_JS.index("function setStatus(")
    script = (
        _PAGE_JS[start:end]
        + "\nconst inputs = ['<img src=x onerror=alert(1)>', '\"><script>x</script>',"
        " \"it's & <b>\", null, 42];\n"
        "console.log(JSON.stringify(inputs.map(esc)));\n"
    )
    runtime = _js_runtime()
    assert runtime is not None
    run = subprocess.run(
        [runtime, "-e", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )

    escaped = json.loads(run.stdout)
    assert escaped == [
        "&lt;img src=x onerror=alert(1)&gt;",
        "&quot;&gt;&lt;script&gt;x&lt;/script&gt;",
        "it&#39;s &amp; &lt;b&gt;",
        "",
        "42",
    ]
    assert not any(ch in value for value in escaped for ch in "<>")


def test_post_writes_output_log_flag(web_base: str, tmp_path: Path) -> None:
    payload = {
        "webhook_server": {"host": "", "port": "", "allowed_hosts": [], "output_log": True}
    }
    status, _ = _post(web_base + "/api/config", payload)
    assert status == 200
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "output_log: true" in written

    status, _, body = _request(web_base + "/api/config")
    assert status == 200
    assert json.loads(body)["data"]["webhook_server"]["output_log"] is True


def test_post_omits_a_false_output_log_flag(web_base: str, tmp_path: Path) -> None:
    payload = {
        "webhook_server": {"host": "0.0.0.0", "port": "", "allowed_hosts": [], "output_log": False}
    }
    status, _ = _post(web_base + "/api/config", payload)
    assert status == 200
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "output_log" not in written
    assert "host: 0.0.0.0" in written


def test_post_rejects_a_non_boolean_output_log(web_base: str, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("agents:\n  - name: a\n", encoding="utf-8")
    status, data = _post(web_base + "/api/config", {"webhook_server": {"output_log": "yes"}})
    assert status == 400
    assert "output_log: expected a boolean" in data["error"]
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == "agents:\n  - name: a\n"


def test_api_never_returns_secret_values_and_keeps_redacted_literals_on_save(
    web_base: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_T_KEY", "resolved-key-value-1234")
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  openai: {source: env:WEB_T_KEY}\n"
        "providers:\n"
        "  a: {api_key: 'secret:openai'}\n"
        "  legacy: {api_key: legacy-literal-key}\n"
        "personas:\n  p: {SOUL.md: 'the key is $WEB_T_KEY'}\n",
        encoding="utf-8",
    )
    status, body = _get(web_base + "/api/config")
    assert status == 200
    assert "resolved-key-value-1234" not in body and "legacy-literal-key" not in body
    data = json.loads(body)["data"]
    assert data["providers"]["a"]["api_key"] == "secret:openai"  # the reference is shown
    shown = data["providers"]["legacy"]["api_key"]
    assert shown == "[REDACTED:providers.legacy.api_key]"

    # Saving the form back as shown keeps the literal on disk instead of the placeholder.
    providers = [{"name": n, **spec} for n, spec in data["providers"].items()]
    status, reply = _post(web_base + "/api/config", {"providers": providers})
    assert status == 200, reply
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["providers"]["legacy"]["api_key"] == "legacy-literal-key"

    # A value pasted as a source is refused.
    status, reply = _post(
        web_base + "/api/config", {"secrets": [{"name": "x", "source": "hunter2hunter2"}]}
    )
    assert status == 400 and "env:NAME or file:PATH" in reply["error"]


def test_secret_value_goes_to_the_file_backend_not_the_config(
    web_base: str, tmp_path: Path
) -> None:
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  forge:\n    source: env:UNSET_X\n"
        "    allow: {tools: [forgejo_pr], hosts: [h]}\n",
        encoding="utf-8",
    )
    status, reply = _post(web_base + "/api/secret", {"name": "forge", "value": "v4lue-v4lue"})
    assert status == 200 and reply == {"ok": True, "source": "file:secrets/forge"}

    stored = tmp_path / "secrets" / "forge"
    assert stored.read_text(encoding="utf-8") == "v4lue-v4lue"
    assert stored.stat().st_mode & 0o777 == 0o600
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "v4lue" not in text
    saved = yaml.safe_load(text)["secrets"]["forge"]
    allow = {"tools": ["forgejo_pr"], "hosts": ["h"]}
    assert saved == {"source": "file:secrets/forge", "allow": allow}
    assert Config.load().secrets.values()["forge"] == "v4lue-v4lue"
    status, body = _get(web_base + "/api/config")
    assert "v4lue" not in body
