"""Tool registry: the set of tools the agent can invoke."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import Tool
from .builtin import build_tool_registry, builtin_tools, default_registry

TOOL_TIMEOUT_SECONDS = 60.0


_SCALAR_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _check_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Check ``arguments`` against a tool's JSON Schema; return a message for the
    model describing the first problem, or ``None`` if they are acceptable.

    Checks required arguments and scalar types (``string``/``integer``/``number``/
    ``boolean``). ``null`` for an optional argument counts as absent. ``array``/
    ``object`` arguments are left to the tool, which accepts several shapes.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    for key in schema.get("required", []):
        if arguments.get(key) is None:
            return f"missing required argument '{key}'"
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        allowed = _SCALAR_TYPES.get(expected) if isinstance(expected, str) else None
        if value is None or allowed is None:
            continue
        # bool is an int subclass in Python, but JSON booleans are not numbers.
        if isinstance(value, bool) != (expected == "boolean") or not isinstance(value, allowed):
            return f"argument '{key}' must be of type {expected}, got {type(value).__name__}"
    return None


class ToolRegistry:
    """A registry of :class:`Tool` instances keyed by name.

    Each tool call is limited to ``timeout`` seconds (async tools only: a tool
    blocking the event loop synchronously cannot be interrupted)."""

    def __init__(self, timeout: float = TOOL_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-compatible "function" tool schemas for all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name. Errors are returned as strings so the model can
        react to them instead of the harness crashing."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        problem = _check_arguments(tool.parameters, arguments)
        if problem is not None:
            return f"Error: invalid arguments for '{name}': {problem}."
        try:
            result = await asyncio.wait_for(tool.run(arguments), timeout=self.timeout)
        except TimeoutError:
            return f"Error: tool '{name}' timed out after {self.timeout:g}s."
        except Exception as exc:  # noqa: BLE001 - report tool errors to the model
            return f"Error: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else str(result)


__all__ = ["Tool", "ToolRegistry", "build_tool_registry", "builtin_tools", "default_registry"]
