"""ToolRegistry behaviour that the built-in tools do not exercise."""

from __future__ import annotations

import asyncio
from typing import Any

from light_harness.tools import ToolRegistry


class _Recorder:
    name = "recorder"
    description = "records its arguments"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {"type": "array"},
            "options": {"type": "object"},
            "label": {"type": "string"},
        },
        "required": ["label"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return "ok"


class _Exploding:
    name = "exploding"
    description = "always raises"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any]) -> str:
        raise ValueError("kaboom")


def _registry(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_array_and_object_arguments_are_passed_through_unchecked() -> None:
    """By design only scalar types are validated: the tool decides what shapes of
    ``array``/``object`` it accepts (``edit_file`` takes a list, an object or JSON text)."""
    recorder = _Recorder()
    registry = _registry(recorder)

    result = asyncio.run(
        registry.run("recorder", {"label": "x", "items": "not a list", "options": [1, 2]})
    )

    assert result == "ok"
    assert recorder.calls == [{"label": "x", "items": "not a list", "options": [1, 2]}]


def test_unknown_arguments_are_not_rejected() -> None:
    recorder = _Recorder()

    result = asyncio.run(_registry(recorder).run("recorder", {"label": "x", "surprise": 1}))

    assert result == "ok"
    assert recorder.calls == [{"label": "x", "surprise": 1}]


def test_a_tool_that_raises_reports_the_error_to_the_model() -> None:
    registry = _registry(_Exploding())

    assert asyncio.run(registry.run("exploding", {})) == "Error: ValueError: kaboom"


def test_an_unknown_tool_name_is_an_error_string() -> None:
    assert asyncio.run(_registry().run("nope", {})) == "Error: unknown tool 'nope'."
