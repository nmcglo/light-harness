"""Per-persona session logs.

Each chat session is written to an append-only JSONL file under
``LIGHT_HARNESS_HOME/personas/<persona>/sessions/``. The first line records
session metadata (persona, agent, model); every subsequent line is one chat
message, so a session file captures the full conversation: user prompts,
assistant replies, the tool calls the model requested, and each tool's result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .llm import ChatMessage


def _utc_stamp() -> str:
    """A sortable, filesystem-safe UTC timestamp (microsecond precision)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


class SessionLog:
    """An append-only JSONL log for a single session under a persona's ``sessions/`` dir.

    Opened in ``__init__`` (which writes the session header line) and must be
    :meth:`closed` when the session ends.
    """

    def __init__(self, sessions_dir: Path, meta: dict[str, str]) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = sessions_dir / f"{_utc_stamp()}_{uuid4().hex[:8]}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps({"type": "session", **meta}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def record(self, message: ChatMessage) -> None:
        """Append one chat message (user, assistant, or tool result) as a line."""
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
        if message.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in message.tool_calls
            ]
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has been called."""
        return self._fh.closed

    def close(self) -> None:
        self._fh.close()
