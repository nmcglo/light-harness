"""Built-in tools.

``echo``, ``current_time`` and ``add`` are trivial examples of the :class:`Tool`
interface. ``read_file``, ``write_file`` and ``edit_file`` do real work on disk and
are confined to an agent's workspace directory (see :mod:`.workspace`); they exist
only when a workspace is configured. Add your own tools the same way and register
them in a :class:`ToolRegistry`.

Their disk work runs in a worker thread (``asyncio.to_thread``) so a slow or
stalled file cannot block the event loop or outrun the registry's timeout.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import config_home
from .base import Tool
from .edit_diff import (
    EditError,
    apply_edits,
    detect_line_ending,
    generate_diff,
    normalize_to_lf,
    restore_line_endings,
    split_bom,
)
from .workspace import Workspace, WorkspaceError, atomic_write_bytes, file_lock

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


_READ_MAX_LINES = 2000
_READ_MAX_BYTES = 50 * 1024

_MAX_FILE_BYTES = 10 * 1024 * 1024


def _check_size(real: Path, label: str) -> None:
    """Refuse files too large to load into memory; raises :class:`WorkspaceError`."""
    size = real.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise WorkspaceError(
            f"file is too large ({_format_size(size)}, limit {_format_size(_MAX_FILE_BYTES)}): "
            f"{label}"
        )


def _format_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable size (B/KB/MB)."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / 1024 / 1024:.1f}MB"


def _truncate_head(
    text: str, max_lines: int, max_bytes: int
) -> tuple[str, bool, str | None, int]:
    """Truncate ``text`` from the head to ``max_lines``/``max_bytes`` (whole lines).

    Returns ``(content, truncated, truncated_by, output_lines)`` where
    ``truncated_by`` is ``"lines"``, ``"bytes"``, or ``None``.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return text, False, None, total_lines
    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return "", True, "bytes", 0
    out: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out.append(line)
        out_bytes += line_bytes
    if len(out) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"
    return "\n".join(out), True, truncated_by, len(out)


class ReadFileTool:
    """Read a file and return its text, optionally a slice of lines.

    Large reads are truncated to a line/byte cap with a hint for how to continue.
    """

    name = "read_file"
    description = (
        "Read a file from disk and return its text. Optionally start at offset "
        "(1-based line number) and/or return at most limit lines. Large reads are "
        "truncated to a line/byte cap and end with a hint for how to continue."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the workspace root.",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "1-based start line (default 1).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of lines to return (1 or more).",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        # Everything below touches the disk: keep the event loop free so the
        # registry's timeout can actually fire.
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> str:
        path = str(arguments["path"])
        raw_offset = arguments.get("offset")
        offset = 1 if raw_offset is None else int(raw_offset)
        raw_limit = arguments.get("limit")
        limit = None if raw_limit is None else int(raw_limit)
        if offset < 1:
            return f"Error: read_file offset must be 1 or more, got {offset}."
        if limit is not None and limit < 1:
            return f"Error: read_file limit must be 1 or more, got {limit}."
        start = offset - 1
        try:
            real = self._workspace.resolve(path)
            _check_size(real, path)
            with open(real, encoding="utf-8") as f:
                text = f.read()
        except WorkspaceError as exc:
            return f"Error: {exc}"
        except FileNotFoundError:
            return f"Error: no such file: {path}"
        except IsADirectoryError:
            return f"Error: is a directory: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"
        except UnicodeDecodeError:
            return f"Error: {path} is not valid UTF-8 text."

        if not text:
            return ""
        all_lines = text.split("\n")
        # A newline-terminated file splits into one trailing empty segment that is not a
        # line: it is not counted, but a slice reaching it keeps the terminator.
        total = len(all_lines) - (1 if text.endswith("\n") else 0)
        if start >= total:
            return f"Offset {offset} is beyond end of file ({total} lines total)."
        start_display = start + 1
        end = len(all_lines) if limit is None else min(start + limit, len(all_lines))
        selected = "\n".join(all_lines[start:end])

        content, truncated, truncated_by, output_lines = _truncate_head(
            selected, _READ_MAX_LINES, _READ_MAX_BYTES
        )
        if truncated and output_lines == 0:
            first_bytes = len(all_lines[start].encode("utf-8"))
            return (
                f"Line {start_display} is {_format_size(first_bytes)}, exceeds the "
                f"{_format_size(_READ_MAX_BYTES)} limit. Use a smaller limit or a narrower range."
            )
        if truncated:
            # output_lines can include the trailing empty segment; no line exists past
            # `total`, so never hint at an offset the file does not have.
            end_display = min(start_display + output_lines - 1, total)
            limit_note = (
                f" ({_format_size(_READ_MAX_BYTES)} limit)" if truncated_by == "bytes" else ""
            )
            tail = f" Use offset={end_display + 1} to continue." if end_display < total else ""
            return (
                f"{content}\n\n[Showing lines {start_display}-{end_display} of {total}"
                f"{limit_note}.{tail}]"
            )
        if limit is not None and end < total:
            return (
                f"{content}\n\n[{total - end} more lines in file. "
                f"Use offset={end + 1} to continue.]"
            )
        return content


class WriteFileTool:
    """Create or overwrite a file. Use only for new files or complete rewrites."""

    name = "write_file"
    description = (
        "Create or overwrite a file. Automatically creates parent directories. "
        "Use write only for new files or complete rewrites; prefer edit_file for "
        "partial changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write, relative to the workspace root.",
            },
            "content": {"type": "string", "description": "Full content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        # Everything below touches the disk: keep the event loop free so the
        # registry's timeout can actually fire.
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> str:
        path = str(arguments["path"])
        content = arguments.get("content") or ""
        data = content.encode("utf-8")
        if len(data) > _MAX_FILE_BYTES:
            return f"Error: content exceeds the {_format_size(_MAX_FILE_BYTES)} limit."
        try:
            real = self._workspace.resolve(path)
            real.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(real):
                atomic_write_bytes(real, data)
        except WorkspaceError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Wrote {len(content)} characters to {path}."


def _prepare_edits(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize the accepted edit argument forms into ``(old_text, new_text)`` pairs.

    Accepts ``edits`` as a list, a single ``{old_text, new_text}`` object, or a JSON
    string, plus a top-level ``old_text``/``new_text`` shorthand. Raises
    ``ValueError`` naming the problem when an edit was sent but is malformed, rather
    than dropping it.
    """
    edits = arguments.get("edits")
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except (ValueError, TypeError):
            raise ValueError("'edits' is a string that is not valid JSON") from None
    collected: list[tuple[str, str]] = []
    if isinstance(edits, dict):
        edits = [edits]
    if edits is not None and not isinstance(edits, list):
        raise ValueError("'edits' must be a list of {old_text, new_text} objects")
    for index, entry in enumerate(edits or []):
        collected.append(_edit_pair(entry, f"edits[{index}]"))
    if "old_text" in arguments or "new_text" in arguments:
        collected.append(_edit_pair(arguments, "old_text/new_text"))
    return collected


def _edit_pair(entry: Any, where: str) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be an object with old_text and new_text")
    old = entry.get("old_text")
    new = entry.get("new_text")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError(f"{where} needs string old_text and new_text")
    return old, new


class EditFileTool:
    """Apply one or more exact-text replacements to a file.

    Mirrors Pi's ``edit`` tool: every edit is matched against the original file,
    each ``old_text`` must be unique and non-overlapping, and the BOM, the original
    line endings, and unchanged lines are preserved. Returns a diff of the changes.
    """

    name = "edit_file"
    description = (
        "Edit a file using one or more exact-text replacements. Provide edits as a "
        "list of {old_text, new_text} pairs; each old_text must be unique and must "
        "not overlap the others. A single {old_text, new_text} pair is accepted as a "
        "shorthand. Returns a diff of the changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit, relative to the workspace root.",
            },
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to find (must be unique).",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                    },
                    "required": ["old_text", "new_text"],
                },
            },
            "old_text": {
                "type": "string",
                "description": "Shorthand: text to find for a single edit.",
            },
            "new_text": {
                "type": "string",
                "description": "Shorthand: replacement text for old_text.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        # Everything below touches the disk, and the diff is CPU-bound: keep the
        # event loop free so the registry's timeout can actually fire.
        return await asyncio.to_thread(self._run_blocking, arguments)

    def _run_blocking(self, arguments: dict[str, Any]) -> str:
        path = str(arguments["path"])
        try:
            edits = _prepare_edits(arguments)
        except ValueError as exc:
            return f"Error: edit_file: {exc}."
        if not edits:
            return (
                "Error: edit_file requires at least one replacement "
                "(edits, or old_text/new_text)."
            )
        try:
            real = self._workspace.resolve(path)
        except WorkspaceError as exc:
            return f"Error: {exc}"
        try:
            # The lock spans read → match → write, so a concurrent edit cannot be
            # overwritten with a result computed from stale content.
            with file_lock(real):
                return self._edit_locked(real, path, edits)
        except FileNotFoundError:
            return f"Error: no such file: {path}"
        except IsADirectoryError:
            return f"Error: is a directory: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"
        except OSError as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def _edit_locked(self, real: Path, path: str, edits: list[tuple[str, str]]) -> str:
        try:
            _check_size(real, path)
            raw_bytes = real.read_bytes()
        except WorkspaceError as exc:
            return f"Error: {exc}"
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: {path} is not valid UTF-8 text."
        try:
            bom, content = split_bom(raw)
            ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base, new = apply_edits(normalized, edits, path)
            final = bom + restore_line_endings(new, ending)
            data = final.encode("utf-8")
            if len(data) > _MAX_FILE_BYTES:
                return f"Error: edited file would exceed the {_format_size(_MAX_FILE_BYTES)} limit."
            atomic_write_bytes(real, data)
        except EditError as exc:
            return f"Error: {exc}"
        diff = generate_diff(base, new, path)
        return f"Successfully replaced {len(edits)} block(s) in {path}.\n{diff}"


BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    EchoTool.name,
    CurrentTimeTool.name,
    AddTool.name,
    ReadFileTool.name,
    WriteFileTool.name,
    EditFileTool.name,
)
FILE_TOOL_NAMES: frozenset[str] = frozenset(
    {ReadFileTool.name, WriteFileTool.name, EditFileTool.name}
)


def make_workspace(root: Path) -> Workspace:
    """A :class:`Workspace` at ``root`` that also keeps the harness's own state
    directory (``config.yaml`` with API keys, session logs) off limits."""
    return Workspace(root, deny=[config_home()])


def builtin_tools(workspace: Path | None = None) -> list[Tool]:
    """The built-in tools, in registration order.

    The file tools exist only when ``workspace`` is given: without a root to
    confine them to they would have the whole filesystem.
    """
    tools: list[Tool] = [EchoTool(), CurrentTimeTool(), AddTool()]
    if workspace is not None:
        ws = make_workspace(workspace)
        tools += [ReadFileTool(ws), WriteFileTool(ws), EditFileTool(ws)]
    return tools


def default_registry(workspace: Path | None = None) -> ToolRegistry:
    """Build a registry preloaded with every built-in tool available for ``workspace``."""
    from . import ToolRegistry  # local import avoids a circular import at load

    registry = ToolRegistry()
    for tool in builtin_tools(workspace):
        registry.register(tool)
    return registry


def build_tool_registry(
    selection: list[str] | None, workspace: Path | None = None
) -> ToolRegistry:
    """Build a registry from an opt-in tool selection.

    ``None`` selects every built-in tool available (the file tools only when a
    ``workspace`` is given); a list selects only the named ones (unknown names, and
    file tools without a workspace, are ignored); an empty list yields no tools.
    """
    from . import ToolRegistry  # local import avoids a circular import at load

    if selection is None:
        return default_registry(workspace)
    registry = ToolRegistry()
    wanted = set(selection)
    for tool in builtin_tools(workspace):
        if tool.name in wanted:
            registry.register(tool)
    return registry
