"""Tests for per-persona session logs (append-only JSONL)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from light_harness.agent import Agent
from light_harness.llm import ChatMessage, LLMResponse, ToolCall
from light_harness.sessions import SessionLog

from ._helpers import sample_registry


class FakeClient:
    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script

    async def chat(self, messages, tools=None):
        return self.script.pop(0)

    async def close(self) -> None:
        return None


def _read_log(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_session_log_records_messages(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session = SessionLog(session_dir, meta={"persona": "default", "model": "m"})
    session.record(ChatMessage(role="user", content="hi"))
    session.record(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call-1", name="add", arguments={"a": 2, "b": 3})],
        )
    )
    session.record(ChatMessage(role="tool", tool_call_id="call-1", content="5"))
    session.close()

    logs = sorted(session_dir.glob("*.jsonl"))
    assert len(logs) == 1
    entries = _read_log(logs[0])
    assert entries[0] == {"type": "session", "persona": "default", "model": "m"}
    assert [e["role"] for e in entries[1:]] == ["user", "assistant", "tool"]
    assert entries[2]["tool_calls"] == [
        {"id": "call-1", "name": "add", "arguments": {"a": 2, "b": 3}}
    ]
    assert entries[3]["tool_call_id"] == "call-1"
    assert entries[3]["content"] == "5"


def test_agent_records_full_session_to_log(tmp_path: Path) -> None:
    client = FakeClient(
        [
            LLMResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="add", arguments={"a": 2, "b": 3})],
                ),
                tool_calls=[ToolCall(id="call-1", name="add", arguments={"a": 2, "b": 3})],
            ),
            LLMResponse(
                message=ChatMessage(role="assistant", content="The sum is 5."),
                tool_calls=[],
            ),
        ]
    )
    session_dir = tmp_path / "sessions"
    session = SessionLog(session_dir, meta={"persona": "default", "agent": "a", "model": "m"})
    agent = Agent(client, sample_registry(), session=session)
    answer = asyncio.run(agent.run("What is 2 + 3?"))
    session.close()

    assert answer == "The sum is 5."
    logs = sorted(session_dir.glob("*.jsonl"))
    assert len(logs) == 1
    entries = _read_log(logs[0])
    assert entries[0]["type"] == "session"
    assert [e["role"] for e in entries[1:]] == ["user", "assistant", "tool", "assistant"]
    # The assistant turn that requested the tool call captured its call + arguments.
    assert entries[2]["tool_calls"] == [
        {"id": "call-1", "name": "add", "arguments": {"a": 2, "b": 3}}
    ]
    # The tool result line carries the tool_call_id and the tool's output.
    assert entries[3]["tool_call_id"] == "call-1"
    assert entries[3]["content"] == "5"
    # The final assistant answer is the last line.
    assert entries[4]["content"] == "The sum is 5."
