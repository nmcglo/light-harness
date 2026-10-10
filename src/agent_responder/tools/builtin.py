"""Built-in example tools.

These are intentionally trivial and demonstrate the :class:`Tool` interface.
Add your own tools the same way and register them in a :class:`ToolRegistry`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .base import Tool

if TYPE_CHECKING:
    from . import ToolRegistry


class EchoTool:
    """Return the provided text unchanged."""

    name = "echo"
    description = "Return the provided text unchanged."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["text"])


class CurrentTimeTool:
    """Return the current UTC time as an ISO-8601 string."""

    name = "current_time"
    description = "Return the current UTC time as an ISO-8601 string."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any]) -> str:
        return datetime.now(UTC).isoformat()


class AddTool:
    """Add two numbers and return their sum."""

    name = "add"
    description = "Add two numbers and return their sum."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])


def builtin_tools() -> list[Tool]:
    """The built-in tools, in registration order."""
    return [EchoTool(), CurrentTimeTool(), AddTool()]


def default_registry() -> ToolRegistry:
    """Build a registry preloaded with all built-in tools."""
    from . import ToolRegistry  # local import avoids a circular import at load

    registry = ToolRegistry()
    for tool in builtin_tools():
        registry.register(tool)
    return registry


def build_tool_registry(selection: list[str] | None) -> ToolRegistry:
    """Build a registry from an opt-in tool selection.

    ``None`` selects all built-in tools; a list selects only the named built-in
    tools (unknown names are ignored); an empty list yields no tools.
    """
    from . import ToolRegistry  # local import avoids a circular import at load

    if selection is None:
        return default_registry()
    registry = ToolRegistry()
    wanted = set(selection)
    for tool in builtin_tools():
        if tool.name in wanted:
            registry.register(tool)
    return registry
