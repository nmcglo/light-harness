"""Workspace confinement for the file tools.

A :class:`Workspace` is a root directory the file tools are allowed to touch.
Every path the model supplies is resolved (following symlinks) and must land
inside the root — never the root itself, which is a directory — and outside any
``deny`` directory, so ``../`` traversal, absolute paths and symlinks pointing
out of the root are all refused.

Limitation: the check and the later open are separate steps, so someone else on
the machine who can swap a directory for a symlink in between could still escape.
An existing path must also be a *regular* file: a FIFO, socket or device reports
``st_size`` 0, so the size limit would not stop it, and reading one blocks the
tool forever. A file with more than one hard link is refused too, since its other
names may be outside the workspace or in a ``deny`` directory. A ``deny`` directory
is matched by identity (device and inode) as well as by path, so a different
spelling of it on a case-insensitive filesystem is still refused. The tools never
create symlinks themselves.
"""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]


class WorkspaceError(Exception):
    """A path the tools must refuse; the message is shown to the model."""


class Workspace:
    """A root directory plus directories inside/around it that stay off limits."""

    def __init__(self, root: Path, deny: Iterable[Path] = ()) -> None:
        self.root = root.resolve()
        self.deny = tuple(d.resolve() for d in deny)

    def resolve(self, raw: str) -> Path:
        """The real path for ``raw`` (relative to the root unless absolute), or
        raise :class:`WorkspaceError` if it is outside the workspace, denied, the
        root directory itself, or an existing path that is not a regular file
        (directory, FIFO, socket, device). A path that does not exist yet is
        returned as-is so ``write_file`` can create it."""
        if not raw or "\x00" in raw:
            raise WorkspaceError("invalid path")
        try:
            candidate = (self.root / raw).resolve()
        except (OSError, RuntimeError) as exc:  # e.g. a symlink loop
            raise WorkspaceError(f"cannot resolve path {raw!r}: {exc}") from exc
        if candidate == self.root:
            raise WorkspaceError(
                f"path is the workspace root directory, not a file: {raw}"
            )
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path is outside the workspace: {raw}")
        if self._is_denied(candidate):
            raise WorkspaceError(f"path is not accessible: {raw}")
        try:
            info = candidate.stat()
        except FileNotFoundError:
            return candidate  # write_file is allowed to create it
        mode = info.st_mode
        if stat.S_ISDIR(mode):
            raise WorkspaceError(f"path is a directory, not a file: {raw}")
        if not stat.S_ISREG(mode):
            raise WorkspaceError(f"path is not a regular file (FIFO, socket or device): {raw}")
        if info.st_nlink > 1:
            # Another name for this inode may live anywhere, including in a denied
            # directory, and ``resolve()`` cannot see it. The tools never create
            # hard links, and an atomic write replaces the file, so none are expected.
            raise WorkspaceError(f"path has multiple hard links: {raw}")
        return candidate

    def resolve_directory(self, raw: str) -> Path:
        """The real path for a directory ``raw`` that a tool may create or fill.

        Same confinement as :meth:`resolve` (inside the root, never the root itself,
        never denied), but the target may be missing or an existing *directory*;
        an existing non-directory is refused."""
        if not raw or "\x00" in raw:
            raise WorkspaceError("invalid path")
        try:
            candidate = (self.root / raw).resolve()
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError(f"cannot resolve path {raw!r}: {exc}") from exc
        if candidate == self.root:
            raise WorkspaceError(f"path is the workspace root directory itself: {raw}")
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path is outside the workspace: {raw}")
        if self._is_denied(candidate):
            raise WorkspaceError(f"path is not accessible: {raw}")
        if candidate.exists() and not candidate.is_dir():
            raise WorkspaceError(f"path exists and is not a directory: {raw}")
        return candidate

    def _is_denied(self, candidate: Path) -> bool:
        """Whether ``candidate`` is, or lies inside, a denied directory.

        The lexical test is not enough on a case-insensitive or
        normalization-insensitive filesystem (``.LIGHT-HARNESS`` is the same
        directory as ``.light-harness``), so every existing ancestor is also
        compared by identity (device and inode) with each denied directory."""
        if any(candidate.is_relative_to(d) for d in self.deny):
            return True
        for ancestor in (candidate, *candidate.parents):
            for denied in self.deny:
                try:
                    if ancestor.samefile(denied):
                        return True
                except OSError:  # a missing ancestor or deny directory has no identity
                    continue
        return False


def _mkstemp_respecting_umask(directory: Path, prefix: str, suffix: str) -> tuple[int, str]:
    """Create a unique temp file in ``directory`` opened for writing. Unlike
    :func:`tempfile.mkstemp`, the mode is plain ``0o666`` narrowed by the process
    umask — what a normal ``open()`` would produce — so it never needs to be
    widened and is never broader than the final file."""
    while True:
        tmp = directory / f"{prefix}{os.urandom(8).hex()}{suffix}"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        return fd, str(tmp)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory and an
    atomic rename, so a crash never leaves a truncated file. The data is fsynced
    before the rename and the directory after it, so a power loss leaves either the
    old or the new content rather than an empty file. An existing file's
    permission bits are kept; a new file gets ``0o666`` narrowed by the umask,
    like any normal file creation."""
    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None
    fd, tmp = _mkstemp_respecting_umask(path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Make the rename durable. Best effort: some filesystems and platforms cannot
    open or fsync a directory, and the file itself is already safely written."""
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock while a file is read-modified-written.

    The lock is ``flock`` on the file's *directory*, not the file: writes replace
    the file with a new inode, which would orphan a lock held on the old one, and a
    sidecar lock file would litter the workspace. ``flock`` conflicts between
    separate open descriptions, so it serializes threads of one process and
    separate processes alike (``agent chat`` plus a webhook run). It is advisory,
    so it only orders cooperating writers, and it is coarser than needed: edits to
    different files in one directory take turns. A no-op where ``fcntl`` is missing.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing releases the lock
