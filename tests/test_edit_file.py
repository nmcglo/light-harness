"""``edit_file`` must refuse an ``old_text`` that cannot locate a real region."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from light_harness.tools import build_tool_registry

ORIGINAL = "hello world\nsecond\tline\n"


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def _run(workspace: Path, **args: object) -> str:
    registry = build_tool_registry(None, workspace)
    return asyncio.run(registry.run("edit_file", dict(args)))


@pytest.mark.parametrize(
    "old_text",
    ["", " ", "  ", "\t", "\u00a0", "\u2003", " \t "],
)
def test_blank_old_text_is_refused_and_changes_nothing(ws: Path, old_text: str) -> None:
    """A whitespace-only pattern fuzzy-matches to nothing: it used to "match" at
    byte 0 with zero length and prepend ``new_text`` to the file."""
    target = ws / "notes.txt"
    target.write_text(ORIGINAL)

    result = _run(ws, path="notes.txt", old_text=old_text, new_text="X")

    assert "Error:" in result
    assert "must not be empty or whitespace-only" in result
    assert target.read_text() == ORIGINAL
    # No temp file left behind by the refused write.
    assert [p.name for p in ws.iterdir()] == ["notes.txt"]


def test_one_blank_edit_rejects_the_whole_batch(ws: Path) -> None:
    target = ws / "notes.txt"
    target.write_text(ORIGINAL)

    result = _run(
        ws,
        path="notes.txt",
        edits=[{"old_text": "second", "new_text": "third"}, {"old_text": " ", "new_text": "X"}],
    )

    assert "Error:" in result
    assert "edits[1].old_text must not be empty or whitespace-only" in result
    assert target.read_text() == ORIGINAL


def test_whitespace_inside_a_pattern_still_matches(ws: Path) -> None:
    target = ws / "notes.txt"
    target.write_text(ORIGINAL)

    result = _run(ws, path="notes.txt", old_text="second\tline", new_text="third line")

    assert "Successfully replaced" in result
    assert target.read_text() == "hello world\nthird line\n"
