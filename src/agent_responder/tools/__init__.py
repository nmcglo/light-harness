"""Tool registry: the set of tools the agent can invoke."""

from __future__ import annotations

from typing import Any

from .base import Tool
from .builtin import default_registry


class ToolRegistry:
    """A registry of :class:`Tool` instances keyed by name."""

    def __init__(self) -> None:
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
        try:
            result = await tool.run(arguments)
        except Exception as exc:  # noqa: BLE001 - report tool errors to the model
            return f"Error: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else str(result)


__all__ = ["Tool", "ToolRegistry", "default_registry"]
