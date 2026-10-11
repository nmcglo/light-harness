"""Webhook listeners: each configured path triggers a specific agent.

``agent webhooks`` starts a small HTTP server (stdlib ``http.server``, matching
``web.py``) that routes POSTs to the paths named in the ``webhooks:`` config
section. Each webhook hands the parsed request body to its configured agent and
returns the agent's answer. A webhook can optionally require a bearer secret in
the ``Authorization`` header.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .agent import Agent
from .config import AgentConfig, Config, WebhookConfig, ensure_persona, persona_sessions_dir
from .llm import OpenAICompatibleClient
from .sessions import SessionLog
from .tools import build_tool_registry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


def parse_webhook_body(body: bytes) -> str:
    """The agent-facing text for a webhook body.

    A JSON body is re-serialized (pretty) so the agent sees the parsed
    structure; a non-JSON body is passed through as text; an empty body yields
    an empty string.
    """
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    return json.dumps(parsed, indent=2)


def _bearer_token(authorization: str | None) -> str:
    """The bearer token from an ``Authorization`` header, else ``""``."""
    if authorization is None:
        return ""
    parts = authorization.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return ""


def _bearer_matches(secret: str | None, authorization: str | None) -> bool:
    """True when ``secret`` is unset, or the header carries it as a bearer."""
    if secret is None:
        return True
    return hmac.compare_digest(_bearer_token(authorization), secret)


def default_agent_factory(
    config: Config, agent_cfg: AgentConfig, webhook: WebhookConfig
) -> Agent:
    """Build the real agent for a webhook's configured agent (client + tools +
    a persona session log under the agent's persona home). The webhook's own
    ``system_prompt`` (when set) overrides the agent's persona-based prompt."""
    settings = config.endpoint_for(agent_cfg)
    registry = build_tool_registry(agent_cfg.tools)
    system_prompt = webhook.system_prompt or config.system_prompt_for(agent_cfg)
    persona_name = agent_cfg.persona or agent_cfg.name
    ensure_persona(persona_name)
    session = SessionLog(
        persona_sessions_dir(persona_name),
        meta={
            "persona": persona_name,
            "agent": agent_cfg.name,
            "model": settings.model,
            "source": "webhook",
        },
    )
    return Agent(
        client=OpenAICompatibleClient(settings),
        tools=registry,
        system_prompt=system_prompt,
        session=session,
    )


class WebhookHandler(BaseHTTPRequestHandler):
    """Route POSTs to configured webhook paths and run the target agent."""

    config: Config

    def do_GET(self) -> None:
        self._send_json(405, {"message": "webhooks accept POST only"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        webhook = self.config.webhook_by_path(path)
        if webhook is None:
            self._send_json(404, {"message": f"no webhook for path '{path}'"})
            return
        if not _bearer_matches(webhook.secret, self.headers.get("Authorization")):
            self._send_json(401, {"message": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        # Ack immediately: confirm the agent can start, then run it in the
        # background so the sender isn't held open while the model works.
        try:
            agent = self._make_agent(webhook)
        except Exception as exc:  # noqa: BLE001 - report the failure to the caller
            self._send_json(500, {"message": f"{type(exc).__name__}: {exc}"})
            return
        self._spawn(agent, body)
        self._send_json(200, {"agent": webhook.agent, "message": "OK"})

    def _spawn(self, agent: Agent, body: bytes) -> None:
        """Run ``agent`` in the background; the caller already has its ack."""

        def worker() -> None:
            try:
                asyncio.run(self._run(agent, body))
            except Exception as exc:  # noqa: BLE001 - no caller left to report to
                print(f"webhook run failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        threading.Thread(target=worker, daemon=True).start()

    async def _run(self, agent: Agent, body: bytes) -> str:
        try:
            return await agent.run(parse_webhook_body(body))
        finally:
            await agent.client.close()
            if agent.session is not None:
                agent.session.close()

    def _make_agent(self, webhook: WebhookConfig) -> Agent:
        """Build the agent for ``webhook`` (overridden per server)."""
        raise NotImplementedError

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the console clean while the server runs."""


def create_webhook_server(
    config: Config,
    host: str,
    port: int,
    agent_factory: Callable[[Config, AgentConfig, WebhookConfig], Agent] | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) the webhook server for ``config``."""
    factory: Callable[[Config, AgentConfig, WebhookConfig], Agent] = (
        default_agent_factory if agent_factory is None else agent_factory
    )
    bound_config = config

    class BoundHandler(WebhookHandler):
        config = bound_config

        def _make_agent(self, webhook: WebhookConfig) -> Agent:
            agent_cfg = self.config.get_agent(webhook.agent)
            if agent_cfg is None:
                raise LookupError(
                    f"webhook '{webhook.path}' references unknown agent '{webhook.agent}'"
                )
            return factory(self.config, agent_cfg, webhook)

    return ThreadingHTTPServer((host, port), BoundHandler)


def unauthenticated_webhooks(config: Config) -> list[WebhookConfig]:
    """Webhooks that accept requests without credentials (no or empty ``secret``).

    An empty secret is included because ``_bearer_matches`` accepts a request
    with no bearer token against it.
    """
    return [w for w in config.webhooks if not w.secret]


def unauthenticated_warning(config: Config) -> str | None:
    """A warning naming how many webhooks are unauthenticated, or ``None``."""
    open_hooks = unauthenticated_webhooks(config)
    if not open_hooks:
        return None
    paths = ", ".join(w.path for w in open_hooks)
    return (
        f"WARNING: {len(open_hooks)} of {len(config.webhooks)} webhook(s) are "
        f"unauthenticated (no secret) and can be triggered by anyone who can reach "
        f"this server: {paths}"
    )


def serve_webhooks(
    config: Config,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Run the webhook server until interrupted (blocking)."""
    server = create_webhook_server(config, host, port)
    print(f"Serving {len(config.webhooks)} webhook(s) at http://{host}:{port}/")
    warning = unauthenticated_warning(config)
    if warning:
        print(warning, file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
