"""Small helpers shared by the HTTP server tests."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from typing import Any

from light_harness.tools import ToolRegistry
from light_harness.tools.builtin import CurrentTimeTool


class EchoTool:
    """Test-only tool: returns ``text`` unchanged."""

    name = "echo"
    description = "Return the provided text unchanged."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["text"])


class AddTool:
    """Test-only tool: returns ``a + b``."""

    name = "add"
    description = "Add two numbers and return their sum."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])


def sample_registry() -> ToolRegistry:
    """A registry of simple tools (``echo``, ``add``, ``current_time``) for agent-loop tests."""
    registry = ToolRegistry()
    for tool in (EchoTool(), AddTool(), CurrentTimeTool()):
        registry.register(tool)
    return registry


def join_thread(thread: threading.Thread, timeout: float = 5.0) -> None:
    """Join a server thread and fail if it is still running: ``join`` alone returns
    silently on timeout, so a wedged handler thread would otherwise pass."""
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "server thread did not stop within the timeout"


def stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    join_thread(thread)
