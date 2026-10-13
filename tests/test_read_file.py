"""``read_file`` must validate ``offset``/``limit`` and count lines as ``wc -l`` does."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from light_harness.tools import build_tool_registry

SIX_LINES = "1\n2\n3\n4\n5\n6\n"


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def _read(workspace: Path, **args: object) -> str:
    registry = build_tool_registry(None, workspace)
    return asyncio.run(registry.run("read_file", dict(args)))


@pytest.mark.parametrize("limit", [0, -1, -3, -100])
def test_non_positive_limit_is_refused(ws: Path, limit: int) -> None:
    """A negative limit selected nothing and then advertised six remaining lines at a
    negative offset — the model acts on that hint, so it must be refused outright."""
    (ws / "notes.txt").write_text(SIX_LINES)

    result = _read(ws, path="notes.txt", limit=limit)

    assert "Error:" in result
    assert "limit must be 1 or more" in result
    assert "more lines" not in result
    assert "offset=" not in result


@pytest.mark.parametrize("offset", [0, -2])
def test_non_positive_offset_is_refused(ws: Path, offset: int) -> None:
    (ws / "notes.txt").write_text(SIX_LINES)

    result = _read(ws, path="notes.txt", offset=offset)

    assert "Error:" in result
    assert "offset must be 1 or more" in result


def test_limit_reports_the_real_remaining_lines(ws: Path) -> None:
    (ws / "notes.txt").write_text(SIX_LINES)

    result = _read(ws, path="notes.txt", offset=1, limit=2)

    assert result == "1\n2\n\n[4 more lines in file. Use offset=3 to continue.]"


def test_hints_walk_the_file_without_gaps_or_overlap(ws: Path) -> None:
    (ws / "notes.txt").write_text(SIX_LINES)

    seen: list[str] = []
    offset = 1
    for _ in range(10):
        result = _read(ws, path="notes.txt", offset=offset, limit=2)
        content, _, hint = result.partition("\n\n[")
        seen.extend(content.split("\n"))
        if not hint:
            break
        offset = int(hint.split("Use offset=", 1)[1].split(" ", 1)[0])
    else:
        pytest.fail("read_file never reached the end of a six-line file")

    assert seen == ["1", "2", "3", "4", "5", "6"]


@pytest.mark.parametrize(
    ("body", "total"),
    [
        ("1\n2\n3\n", 3),  # newline-terminated: the trailing segment is not a line
        ("1\n2\n3", 3),  # no final newline
        ("1\n2\n\n", 3),  # a real blank final line still counts
        ("", 0),
    ],
)
def test_line_count_matches_the_real_lines(ws: Path, body: str, total: int) -> None:
    (ws / "notes.txt").write_text(body)

    assert _read(ws, path="notes.txt") == body  # a whole-file read is verbatim
    beyond = _read(ws, path="notes.txt", offset=total + 1)
    if total:
        assert f"Offset {total + 1} is beyond end of file ({total} lines total)." in beyond
    else:
        assert beyond == ""


def test_slice_reaching_the_end_keeps_the_final_newline(ws: Path) -> None:
    (ws / "notes.txt").write_text("1\n2\n3\n")

    assert _read(ws, path="notes.txt", offset=2, limit=99) == "2\n3\n"


def test_line_cap_does_not_hint_past_the_end_of_the_file(ws: Path) -> None:
    """A 2000-line file used to count as 2001 lines and advertise ``offset=2001``."""
    (ws / "big.txt").write_text("".join(f"{n}\n" for n in range(1, 2001)))

    result = _read(ws, path="big.txt")

    assert result.endswith("\n\n[Showing lines 1-2000 of 2000.]")
    assert "Use offset=" not in result


def test_line_cap_hint_points_at_a_line_that_exists(ws: Path) -> None:
    (ws / "big.txt").write_text("".join(f"{n}\n" for n in range(1, 2501)))

    result = _read(ws, path="big.txt")

    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in result
    assert _read(ws, path="big.txt", offset=2001, limit=1).startswith("2001\n")


def test_byte_cap_hint_points_at_a_line_that_exists(ws: Path) -> None:
    line = "x" * 99  # 100 bytes per line including the newline
    (ws / "wide.txt").write_text("".join(f"{line}\n" for _ in range(1000)))  # 100 KB

    result = _read(ws, path="wide.txt")

    assert "of 1000 (50.0KB limit). Use offset=" in result
    offset = int(result.rsplit("Use offset=", 1)[1].split(" to continue", 1)[0])
    assert _read(ws, path="wide.txt", offset=offset, limit=1).startswith(line)
