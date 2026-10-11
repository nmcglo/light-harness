"""Tool protocol.

A tool is any object exposing ``name``, ``description``, ``parameters`` (a JSON
Schema for the arguments) and an async ``run``. The registry serializes these
into OpenAI "function" tool schemas, so the model learns how to call them.
"""

from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    """Callable tool the model can invoke by name."""

    name: str
    description: str
    parameters: dict[str, Any]

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with ``arguments`` and return a string result."""
        ...
