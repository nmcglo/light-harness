"""The tool timeout must be enforceable: no workspace path may block the event
loop, and a stalled disk call must not wedge the registry."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from light_harness.tools import ToolRegistry, builtin_tools
from light_harness.tools.builtin import ReadFileTool
from light_harness.tools.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def _run(workspace: Path, tool: str, *, timeout: float = 5.0, **args: object) -> str:
    registry = ToolRegistry(timeout=timeout)
    for candidate in builtin_tools(workspace):
        registry.register(candidate)
    return asyncio.run(registry.run(tool, dict(args)))


def _poke_writer(path: Path) -> None:
    """Open and close the write end: unblocks a reader that is stuck on the FIFO,
    and fails with ENXIO (harmlessly) if the tool never opened it."""
    with contextlib.suppress(OSError):
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))


@pytest.fixture()
def fifo(ws: Path) -> Iterator[Path]:
    """A FIFO in the workspace: opening it for reading succeeds, but ``read()``
    blocks until a writer appears. A timer pokes a writer in after a second, so a
    tool that ever blocks on it returns instead of wedging the test session."""
    path = ws / "pipe"
    os.mkfifo(path)
    timer = threading.Timer(1.0, _poke_writer, args=(path,))
    timer.daemon = True
    timer.start()
    try:
        yield path
    finally:
        timer.cancel()


def test_read_file_refuses_a_fifo_instead_of_reading_it(ws: Path, fifo: Path) -> None:
    """A FIFO reports ``st_size`` 0, so the size limit let it through and
    ``f.read()`` blocked forever where no timeout could fire."""
    result = _run(ws, "read_file", path="pipe", timeout=2.0)

    assert "not a regular file" in result
    assert fifo.is_fifo()


def test_read_file_refuses_a_symlink_to_a_fifo(ws: Path, fifo: Path) -> None:
    (ws / "link").symlink_to(fifo.name)

    assert "not a regular file" in _run(ws, "read_file", path="link", timeout=2.0)
    assert fifo.is_fifo()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no AF_UNIX on this platform")
def test_read_file_refuses_a_socket(ws: Path) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(ws / "sock"))

        assert "not a regular file" in _run(ws, "read_file", path="sock", timeout=2.0)
    finally:
        server.close()


def test_write_and_edit_refuse_a_fifo_without_replacing_it(ws: Path, fifo: Path) -> None:
    assert "not a regular file" in _run(
        ws, "write_file", path="pipe", content="x", timeout=2.0
    )
    assert "not a regular file" in _run(
        ws, "edit_file", path="pipe", old_text="a", new_text="b", timeout=2.0
    )
    assert fifo.is_fifo()


def test_file_tools_refuse_a_directory(ws: Path) -> None:
    (ws / "sub").mkdir()

    assert "is a directory, not a file" in _run(ws, "read_file", path="sub")
    assert "is a directory, not a file" in _run(ws, "write_file", path="sub", content="x")
    assert "is a directory, not a file" in _run(
        ws, "edit_file", path="sub", old_text="a", new_text="b"
    )
    assert [p.name for p in ws.iterdir()] == ["sub"]


class _StalledWorkspace(Workspace):
    """A workspace whose disk phase blocks until the test releases it: a stand-in
    for a filesystem that stops answering (a stalled mount, a wedged device)."""

    def __init__(self, root: Path, gate: threading.Event) -> None:
        super().__init__(root)
        self._gate = gate

    def resolve(self, raw: str) -> Path:
        # Bounded, so a regressed tool fails the test instead of the session.
        self._gate.wait(30)
        return super().resolve(raw)


def test_a_stalled_disk_call_times_out_without_freezing_the_event_loop(ws: Path) -> None:
    (ws / "notes.txt").write_text("hello\n")
    gate = threading.Event()
    registry = ToolRegistry(timeout=0.2)
    registry.register(ReadFileTool(_StalledWorkspace(ws, gate)))
    ticks = 0

    async def main() -> tuple[str, int]:
        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            result = await registry.run("read_file", {"path": "notes.txt"})
            return result, ticks
        finally:
            gate.set()  # let the worker thread finish before the loop shuts down
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat

    result, ticks = asyncio.run(main())

    assert result == "Error: tool 'read_file' timed out after 0.2s."
    assert ticks >= 5, f"the event loop stalled while the tool was blocked ({ticks} ticks)"


def test_the_registry_recovers_after_a_timed_out_file_call(ws: Path) -> None:
    """A timed-out call is abandoned, not killed: the registry must stay usable."""
    (ws / "notes.txt").write_text("hello\n")
    gate = threading.Event()
    registry = ToolRegistry(timeout=0.2)
    registry.register(ReadFileTool(_StalledWorkspace(ws, gate)))

    async def main() -> tuple[str, str]:
        first = await registry.run("read_file", {"path": "notes.txt"})
        gate.set()
        return first, await registry.run("read_file", {"path": "notes.txt"})

    first, second = asyncio.run(main())

    assert first == "Error: tool 'read_file' timed out after 0.2s."
    assert second == "hello\n"
