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


@pytest.mark.parametrize(
    "edits",
    [
        [{"old_text": "second", "new_text": 5}],
        [{"old_text": "second"}],
        ["not an object"],
        "not json",
    ],
)
def test_malformed_edit_entries_are_reported_not_dropped(ws: Path, edits: object) -> None:
    """An entry with a non-string field used to vanish, so the model was told it
    had sent no replacement at all."""
    target = ws / "notes.txt"
    target.write_text(ORIGINAL)

    result = _run(ws, path="notes.txt", edits=edits)

    assert result.startswith("Error: edit_file:")
    assert "at least one replacement" not in result
    assert target.read_text() == ORIGINAL


def test_concurrent_edits_to_one_file_are_not_lost(ws: Path) -> None:
    """Each edit reads, matches and rewrites the whole file; without a lock two
    overlapping runs both start from the same content and the later write erases
    the earlier one."""
    import threading

    count = 12
    target = ws / "notes.txt"
    target.write_text("".join(f"line{i} old\n" for i in range(count)))
    results: list[str] = []

    def work(i: int) -> None:
        results.append(_run(ws, path="notes.txt", old_text=f"line{i} old", new_text=f"line{i} new"))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all("Successfully replaced" in r for r in results)
    assert target.read_text() == "".join(f"line{i} new\n" for i in range(count))


def test_fuzzy_noop_edit_is_refused_and_leaves_the_file_alone(ws: Path) -> None:
    """With a smart quote in the file, ``old_text == new_text`` matched fuzzily and
    the touched line was rewritten in ASCII, so the guard never fired."""
    target = ws / "notes.txt"
    original = "it\u2019s fine  \nother\n"
    target.write_text(original, encoding="utf-8")

    result = _run(ws, path="notes.txt", old_text="it's fine", new_text="it's fine")

    assert "No changes made" in result
    assert target.read_text(encoding="utf-8") == original


def test_unique_exact_match_is_not_rejected_as_a_fuzzy_duplicate(ws: Path) -> None:
    """``"hello "`` occurs once exactly; only its fuzzy form (no trailing space) is
    ambiguous."""
    target = ws / "notes.txt"
    target.write_text("hello \nhello\n")

    result = _run(ws, path="notes.txt", old_text="hello ", new_text="bye ")

    assert "Successfully replaced" in result
    assert target.read_text() == "bye \nhello\n"
