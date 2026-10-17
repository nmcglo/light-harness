"""The file tools must stay inside their workspace (and out of the harness home)."""

from __future__ import annotations

import asyncio
import os
import stat
import textwrap
from pathlib import Path

import pytest

from light_harness.config import Config
from light_harness.tools import build_tool_registry
from light_harness.tools.workspace import Workspace, WorkspaceError


@pytest.fixture()
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """(workspace, outside, harness home) as sibling directories."""
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    home = tmp_path / "home"
    for d in (workspace, outside, home):
        d.mkdir()
    (outside / "secret.txt").write_text("top secret\n")
    (home / "config.yaml").write_text("api_key: sk-123\n")
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(home))
    return workspace, outside, home


def _run(workspace: Path, tool: str, **args: object) -> str:
    registry = build_tool_registry(None, workspace)
    return asyncio.run(registry.run(tool, dict(args)))


def test_reads_and_writes_inside_workspace(dirs: tuple[Path, Path, Path]) -> None:
    ws, _, _ = dirs
    assert _run(ws, "write_file", path="notes/a.txt", content="hello\n").startswith("Wrote")
    assert (ws / "notes" / "a.txt").read_text() == "hello\n"
    assert _run(ws, "read_file", path="notes/a.txt") == "hello\n"
    assert "Successfully replaced" in _run(
        ws, "edit_file", path="notes/a.txt", old_text="hello", new_text="bye"
    )
    assert (ws / "notes" / "a.txt").read_text() == "bye\n"
    # An absolute path that points inside the workspace is fine.
    assert _run(ws, "read_file", path=str(ws / "notes" / "a.txt")) == "bye\n"


@pytest.mark.parametrize("bad", ["../outside/secret.txt", "notes/../../outside/secret.txt"])
def test_dotdot_traversal_is_refused(dirs: tuple[Path, Path, Path], bad: str) -> None:
    ws, outside, _ = dirs
    assert "outside the workspace" in _run(ws, "read_file", path=bad)
    assert "outside the workspace" in _run(ws, "write_file", path=bad + ".new", content="x")
    assert "outside the workspace" in _run(
        ws, "edit_file", path=bad, old_text="top", new_text="pwned"
    )
    assert (outside / "secret.txt").read_text() == "top secret\n"
    assert not (outside / "secret.txt.new").exists()


def test_absolute_path_outside_is_refused(dirs: tuple[Path, Path, Path]) -> None:
    ws, outside, _ = dirs
    target = str(outside / "secret.txt")
    assert "outside the workspace" in _run(ws, "read_file", path=target)
    assert "outside the workspace" in _run(ws, "write_file", path=target, content="x")
    assert (outside / "secret.txt").read_text() == "top secret\n"


def test_symlinks_cannot_escape_but_internal_ones_work(dirs: tuple[Path, Path, Path]) -> None:
    ws, outside, _ = dirs
    (ws / "real.txt").write_text("inside\n")
    (ws / "link_in.txt").symlink_to(ws / "real.txt")
    (ws / "link_out.txt").symlink_to(outside / "secret.txt")
    (ws / "dir_out").symlink_to(outside, target_is_directory=True)

    assert _run(ws, "read_file", path="link_in.txt") == "inside\n"
    assert "outside the workspace" in _run(ws, "read_file", path="link_out.txt")
    assert "outside the workspace" in _run(ws, "read_file", path="dir_out/secret.txt")
    # Writing through an escaping symlinked directory must not create files outside.
    assert "outside the workspace" in _run(ws, "write_file", path="dir_out/new.txt", content="x")
    assert "outside the workspace" in _run(
        ws, "edit_file", path="link_out.txt", old_text="top", new_text="pwned"
    )
    assert not (outside / "new.txt").exists()
    assert (outside / "secret.txt").read_text() == "top secret\n"


def test_harness_home_is_off_limits_even_via_symlink(dirs: tuple[Path, Path, Path]) -> None:
    ws, _, home = dirs
    # Workspace validation rejects roots containing the home, but a symlink inside a
    # valid workspace could still point at it; the tools must refuse that too.
    (ws / "cfg").symlink_to(home, target_is_directory=True)
    assert "outside the workspace" in _run(ws, "read_file", path="cfg/config.yaml")
    nested = home / "workspace"
    nested.mkdir()
    (nested / "x.txt").write_text("x")
    # A workspace *inside* the home is denied by the tool layer as well.
    assert "not accessible" in _run(nested, "read_file", path="x.txt")


def test_write_keeps_the_files_mode(dirs: tuple[Path, Path, Path]) -> None:
    ws, _, _ = dirs
    script = ws / "run.sh"
    script.write_text("old\n")
    script.chmod(0o750)
    _run(ws, "write_file", path="run.sh", content="new\n")
    assert script.read_text() == "new\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o750


@pytest.mark.parametrize("failure", [OSError(28, "No space left on device"), KeyboardInterrupt()])
def test_a_write_that_fails_midway_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """The new content is fully written to a temp file first; if the rename never
    happens (disk full, Ctrl-C) the original must be intact and nothing left behind."""
    from light_harness.tools.workspace import atomic_write_bytes

    target = tmp_path / "f.txt"
    target.write_bytes(b"original")

    def fail(src: object, dst: object) -> None:
        raise failure

    monkeypatch.setattr(os, "replace", fail)

    with pytest.raises(type(failure)):
        atomic_write_bytes(target, b"replacement that never lands")

    monkeypatch.undo()
    assert target.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_a_concurrent_reader_only_ever_sees_a_whole_version(tmp_path: Path) -> None:
    """Readers racing the writer must get one complete version or the other, never an
    empty or half-written file (which a plain truncate-and-write would show)."""
    import threading

    from light_harness.tools.workspace import atomic_write_bytes

    target = tmp_path / "f.txt"
    versions = (b"A" * 400_000, b"B" * 400_000)
    target.write_bytes(versions[0])
    stop = threading.Event()
    seen_bad: list[int] = []

    def reader() -> None:
        while not stop.is_set():
            data = target.read_bytes()
            if data not in versions:
                seen_bad.append(len(data))

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    try:
        for i in range(150):
            atomic_write_bytes(target, versions[i % 2])
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=5)
            assert not t.is_alive()

    assert seen_bad == []


@pytest.mark.parametrize("umask", [0o077, 0o027, 0o022])
def test_new_files_honour_the_umask(
    dirs: tuple[Path, Path, Path], umask: int
) -> None:
    ws, _, _ = dirs
    old_umask = os.umask(umask)
    try:
        assert _run(ws, "write_file", path="new.txt", content="data\n").startswith("Wrote")
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE((ws / "new.txt").stat().st_mode) == 0o666 & ~umask


@pytest.mark.parametrize("bad", [".", "./", "sub/..", "notes/.."])
def test_workspace_root_is_not_a_writable_file(
    dirs: tuple[Path, Path, Path], bad: str
) -> None:
    ws, outside, _ = dirs
    (ws / "notes").mkdir()
    result = _run(ws, "write_file", path=bad, content="escape\n")
    assert result.startswith("Error")
    assert str(ws.parent) not in result  # the refusal must not leak host paths
    # Nothing was created in the workspace's parent, and no temp file inside.
    assert {p.name for p in ws.parent.iterdir()} == {"ws", "outside", "home"}
    assert [p.name for p in ws.iterdir()] == ["notes"]


def test_oversized_files_are_refused(dirs: tuple[Path, Path, Path]) -> None:
    ws, _, _ = dirs
    big = ws / "big.txt"
    with big.open("wb") as fh:
        fh.truncate(11 * 1024 * 1024)  # sparse; not actually written
    assert "too large" in _run(ws, "read_file", path="big.txt")
    assert "exceeds" in _run(ws, "write_file", path="x.txt", content="a" * (11 * 1024 * 1024))


def test_config_fails_closed_on_bad_workspace(
    dirs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    ws, _, home = dirs
    (home / "config.yaml").write_text(
        textwrap.dedent(
            f"""
            agents:
              - {{name: good, workspace: {ws}, tools: [read_file]}}
              - {{name: relative, workspace: some/dir}}
              - {{name: missing, workspace: {tmp_path}/nope}}
              - {{name: wide, workspace: {tmp_path}}}
              - {{name: inside, workspace: {home}}}
              - {{name: nows, tools: [read_file, current_time]}}
            """
        )
    )
    config = Config.load()
    by_name = {a.name: a for a in config.agents}
    assert by_name["good"].workspace == str(ws.resolve())
    for bad in ("relative", "missing", "wide", "inside"):
        assert by_name[bad].workspace is None, bad
    joined = "\n".join(config.warnings)
    assert "absolute" in joined and "existing directory" in joined
    assert "harness home" in joined
    assert "read_file need a 'workspace'" in joined
    # A dropped workspace means no file tools, even with tools unset.
    assert not set(build_tool_registry(None, by_name["wide"].workspace_path()).names()) & {
        "read_file",
        "write_file",
        "edit_file",
    }


def test_hard_linked_file_is_refused(tmp_path: Path) -> None:
    """A second name for a file in the workspace may be a name for a secret outside
    it; ``resolve()`` cannot tell, so such files are not served."""
    secret_dir = tmp_path / "home"
    secret_dir.mkdir()
    secret = secret_dir / "config.yaml"
    secret.write_text("api_key: x")
    root = tmp_path / "ws"
    root.mkdir()
    os.link(secret, root / "innocent.txt")
    (root / "plain.txt").write_text("ok")
    ws = Workspace(root, deny=[secret_dir])

    with pytest.raises(WorkspaceError, match="hard links"):
        ws.resolve("innocent.txt")
    assert ws.resolve("plain.txt") == root.resolve() / "plain.txt"


def test_deny_matches_by_identity_not_spelling(tmp_path: Path) -> None:
    """On a case-insensitive filesystem ``.LIGHT-HARNESS`` is the denied directory
    under another spelling. Emulated with a symlink, which ``resolve()`` has already
    followed for the candidate but which the deny entry never went through."""
    root = tmp_path / "ws"
    root.mkdir()
    home = root / "home"
    home.mkdir()
    (home / "config.yaml").write_text("api_key: x")
    ws = Workspace(root, deny=[home])
    alias = tmp_path / "alias"
    alias.symlink_to(home)
    ws.deny = (alias,)  # spelled differently from the candidate's real path

    with pytest.raises(WorkspaceError, match="not accessible"):
        ws.resolve("home/config.yaml")
    (root / "other.txt").write_text("ok")
    assert ws.resolve("other.txt") == root.resolve() / "other.txt"


def test_atomic_write_syncs_data_before_rename_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is only crash-safe if the new bytes reached the disk first, and
    only durable once the directory entry has been synced."""
    from light_harness.tools.workspace import atomic_write_bytes

    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1]
    )

    atomic_write_bytes(tmp_path / "f.txt", b"data")

    assert events == ["fsync", "replace", "fsync"]
    assert (tmp_path / "f.txt").read_bytes() == b"data"


def test_a_symlink_loop_is_refused_instead_of_crashing(dirs: tuple[Path, Path, Path]) -> None:
    ws, _, _ = dirs
    (ws / "a").symlink_to("b")
    (ws / "b").symlink_to("a")

    with pytest.raises(WorkspaceError, match="cannot resolve path"):
        Workspace(ws).resolve("a")
    assert _run(ws, "read_file", path="a").startswith("Error:")


@pytest.mark.parametrize("raw", ["", "a\x00b", "\x00"])
def test_empty_and_nul_paths_are_invalid(dirs: tuple[Path, Path, Path], raw: str) -> None:
    ws, _, _ = dirs

    with pytest.raises(WorkspaceError, match="invalid path"):
        Workspace(ws).resolve(raw)
