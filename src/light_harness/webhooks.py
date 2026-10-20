"""Webhook listeners: each configured path triggers a specific agent.

``agent webhooks`` starts a small HTTP server (stdlib ``http.server``, matching
``web.py``) that routes POSTs to the paths named in the ``webhooks:`` config
section. Each webhook hands the parsed request body to its configured agent and
acknowledges the POST immediately (200 ``OK``); the agent runs in a background
thread and its answer is not returned to the caller. A webhook can optionally
require a bearer secret in the ``Authorization`` header.

With ``webhook_server.output_log`` (or ``agent webhooks --output-log``), each
accepted call logs itself -- time, path, agent -- and names the session log
file that will hold its run's chat log.

Every request must also carry a ``Host`` header that names this server -- a
loopback name, an IP address, the bind host, or one passed as ``allowed_hosts``
(see :func:`light_harness.htputil.host_is_allowed`). A webhook that needs no
secret is otherwise triggerable by any page that can reach the port, including a
page whose own hostname has been made to resolve to this machine.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .config import (
    AgentConfig,
    Config,
    WebhookConfig,
    config_home,
    ensure_persona,
    persona_sessions_dir,
)
from .httputil import (
    BodyError,
    declared_body,
    host_allow_list,
    host_is_allowed,
    read_body,
    secret_matches,
)
from .llm import OpenAICompatibleClient
from .secrets import Redactor, with_secret_note
from .sessions import SessionLog
from .tools import build_tool_registry
from .tools.builtin import BUILTIN_TOOL_NAMES, WORKSPACE_TOOL_NAMES

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_MAX_CONCURRENT_RUNS = 4
REQUEST_TIMEOUT_SECONDS = 30.0


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
    return secret_matches(_bearer_token(authorization), secret)


async def _release_agent(agent: Agent) -> None:
    """Close an agent's HTTP client and session log; closing the session is
    attempted even if closing the client fails."""
    try:
        await agent.client.close()
    finally:
        if agent.session is not None:
            agent.session.close()


def _session_log_name(path: Path) -> str:
    """A session log file's path relative to the harness home (where session
    logs live); the absolute path if it is not under home."""
    try:
        return str(path.relative_to(config_home()))
    except ValueError:
        return str(path)



def default_agent_factory(
    config: Config, agent_cfg: AgentConfig, webhook: WebhookConfig
) -> Agent:
    """Build the real agent for a webhook's configured agent (client + tools +
    a persona session log under the agent's persona home). The webhook's own
    ``system_prompt`` (when set) overrides the agent's persona-based prompt.

    Webhooks are non-interactive, so secrets with ``confirm`` are denied, and the
    payload is untrusted from the start (``deny_after_untrusted`` secrets too)."""
    settings = config.endpoint_for(agent_cfg)
    secrets = config.tool_secrets(agent_cfg, "webhook")
    secrets.context.untrusted = True
    registry = build_tool_registry(agent_cfg.tools, agent_cfg.workspace_path(), secrets)
    redactor = config.redactor()
    system_prompt = with_secret_note(
        webhook.system_prompt or config.system_prompt_for(agent_cfg), secrets, registry.names()
    )
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
        redactor=redactor,
    )
    try:
        return Agent(
            client=OpenAICompatibleClient(settings, guard=redactor),
            tools=registry,
            system_prompt=system_prompt,
            session=session,
            redactor=redactor,
            secret_context=secrets.context,
        )
    except BaseException:
        session.close()  # don't leak the open log file if construction fails
        raise


class WebhookHandler(BaseHTTPRequestHandler):
    """Route POSTs to configured webhook paths and run the target agent."""

    config: Config
    redactor: Redactor
    run_slots: threading.BoundedSemaphore
    allowed_hosts: frozenset[str]

    # Per-connection socket timeout: a client that stalls mid-request can't hold a
    # handler thread forever.
    timeout = REQUEST_TIMEOUT_SECONDS

    def _method_not_allowed(self) -> None:
        if not self._host_ok():
            return
        self._reject(
            405, {"message": "webhooks accept POST only"}, headers={"Allow": "POST"}
        )

    def _host_ok(self) -> bool:
        """Answer with 403 unless the ``Host`` header names this server."""
        if host_is_allowed(self, self.allowed_hosts):
            return True
        self._reject(403, {"message": "forbidden host"})
        return False

    do_GET = do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _method_not_allowed

    def do_POST(self) -> None:
        if not self._host_ok():
            return
        path = urlsplit(self.path).path
        webhook = self.config.webhook_by_path(path)
        if webhook is None:
            self._reject(404, {"message": f"no webhook for path '{path}'"})
            return
        if not _bearer_matches(webhook.secret, self.headers.get("Authorization")):
            self._reject(401, {"message": "unauthorized"})
            return
        try:
            body = read_body(self)
        except BodyError as exc:
            self._send_json(exc.status, {"message": exc.message})
            return
        # Bound concurrent runs: each accepted POST is an LLM run on its own thread.
        if not self.run_slots.acquire(blocking=False):
            self._send_json(429, {"message": "too many concurrent webhook runs; retry later"})
            return
        # Ack immediately: confirm the agent can start, then run it in the
        # background so the sender isn't held open while the model works.
        agent: Agent | None = None
        try:
            agent = self._make_agent(webhook)
            if self.config.webhook_server.output_log:
                stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                suffix = ""
                if agent.session is not None:
                    suffix = f" (session: {_session_log_name(agent.session.path)})"
                line = f"{stamp} webhook {self.command} {path} -> agent '{webhook.agent}'{suffix}"
                print(self.redactor.redact(line))
            self._spawn(agent, body)
        except Exception as exc:  # noqa: BLE001 - report the failure to the caller
            self.run_slots.release()
            if agent is not None:  # built, but the worker never started
                asyncio.run(_release_agent(agent))
            self._send_json(
                500, {"message": self.redactor.redact(f"{type(exc).__name__}: {exc}")}
            )
            return
        self._send_json(200, {"agent": webhook.agent, "message": "OK"})

    def _spawn(self, agent: Agent, body: bytes) -> None:
        """Run ``agent`` in the background; the caller already has its ack.

        The worker releases the caller's run slot when it finishes."""
        slots = self.run_slots
        redact = self.redactor.redact

        def worker() -> None:
            try:
                asyncio.run(self._run(agent, body))
            except Exception as exc:  # noqa: BLE001 - no caller left to report to
                print(redact(f"webhook run failed: {type(exc).__name__}: {exc}"), file=sys.stderr)
            finally:
                slots.release()

        threading.Thread(target=worker, daemon=True).start()

    async def _run(self, agent: Agent, body: bytes) -> str:
        try:
            text = parse_webhook_body(body)
            secret = agent.secret_in(text)
            if secret is not None:
                print(
                    f"webhook {urlsplit(self.path).path}: payload contains the value of "
                    f"secret '{secret}'; it was redacted before being stored or sent",
                    file=sys.stderr,
                )
            return await agent.run(text)
        finally:
            await _release_agent(agent)

    def _make_agent(self, webhook: WebhookConfig) -> Agent:
        """Build the agent for ``webhook`` (overridden per server)."""
        raise NotImplementedError

    def _reject(
        self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        """Answer a request whose body is not going to be read: the connection has
        to close, otherwise its leftover bytes are parsed as the next request."""
        if declared_body(self):
            self.close_connection = True
        self._send_json(status, payload, headers)

    def _send_json(
        self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":  # HEAD responses carry headers only
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the console clean while the server runs."""


def create_webhook_server(
    config: Config,
    host: str,
    port: int,
    agent_factory: Callable[[Config, AgentConfig, WebhookConfig], Agent] | None = None,
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
    allowed_hosts: Sequence[str] = (),
) -> ThreadingHTTPServer:
    """Build (but do not start) the webhook server for ``config``.

    At most ``max_concurrent_runs`` agent runs are in flight; further POSTs get 429.
    A request whose ``Host`` header does not name this server gets 403 before any
    agent is built; ``allowed_hosts`` are the names the server is really reached by
    (a public hostname, the name a reverse proxy forwards).

    When ``config.webhook_server.output_log`` is set, each accepted call logs
    itself to stdout, naming the session log file of its run."""
    factory: Callable[[Config, AgentConfig, WebhookConfig], Agent] = (
        default_agent_factory if agent_factory is None else agent_factory
    )
    bound_config = config
    bound_slots = threading.BoundedSemaphore(max(1, max_concurrent_runs))
    bound_hosts = host_allow_list(host, allowed_hosts)

    class BoundHandler(WebhookHandler):
        config = bound_config
        redactor = bound_config.redactor()
        run_slots = bound_slots
        allowed_hosts = bound_hosts

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


def tool_availability_warning(config: Config) -> str | None:
    """A warning naming webhooks whose agent cannot use the tools its config implies,
    or ``None``.

    An agent with no ``tools`` list means "every available built-in tool", and the
    workspace tools (the file tools and ``forgejo_clone``) exist only with a workspace:
    a webhook whose agent has none runs without them. Tool names spelled out in ``tools`` are
    already reported by the config loader, so this covers the implied set, plus
    webhook entries naming an agent that does not exist.
    """
    lines: list[str] = []
    for webhook in config.webhooks:
        agent = config.get_agent(webhook.agent)
        if agent is None:
            lines.append(f"{webhook.path}: agent '{webhook.agent}' is not defined")
        elif agent.tools is None and agent.workspace is None:
            implied = [t for t in BUILTIN_TOOL_NAMES if t in WORKSPACE_TOOL_NAMES]
            lines.append(
                f"{webhook.path} (agent '{agent.name}'): no 'workspace', so "
                f"{', '.join(implied)} are unavailable"
            )
    if not lines:
        return None
    return (
        f"WARNING: {len(lines)} webhook(s) cannot use the tools their config implies: "
        + "; ".join(lines)
    )


def _effective_tools(config: Config, webhook: WebhookConfig) -> str:
    """Names of the tools the webhook's agent actually gets, as the run builds them."""
    agent = config.get_agent(webhook.agent)
    if agent is None:
        return "[red]agent not defined[/red]"
    registry = build_tool_registry(agent.tools, agent.workspace_path())
    names = [spec["function"]["name"] for spec in registry.tool_schemas()]
    return ", ".join(names) or "(none)"


def webhook_summary(config: Config, host: str, port: int) -> Table:
    """A table with one row per active webhook: URL, agent, auth, prompt and tools."""
    table = Table(title=f"{len(config.webhooks)} webhook(s)", title_justify="left")
    table.add_column("URL", overflow="fold")
    table.add_column("Agent")
    table.add_column("Auth")
    table.add_column("System prompt")
    table.add_column("Tools", overflow="fold")
    for w in config.webhooks:
        table.add_row(
            f"http://{host}:{port}{w.path}",
            w.agent,
            "bearer secret" if w.secret else "[yellow]none[/yellow]",
            "custom" if w.system_prompt else "agent default",
            _effective_tools(config, w),
        )
    return table


def serve_webhooks(
    config: Config,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allowed_hosts: Sequence[str] = (),
) -> None:
    """Run the webhook server until interrupted (blocking)."""
    server = create_webhook_server(config, host, port, allowed_hosts=allowed_hosts)
    print(f"Webhook server listening on http://{host}:{port}/")
    Console().print(webhook_summary(config, host, port))
    warning = unauthenticated_warning(config)
    if warning:
        print(warning, file=sys.stderr)
    tool_warning = tool_availability_warning(config)
    if tool_warning:
        print(tool_warning, file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
