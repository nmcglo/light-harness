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
tool forever. The tools never create symlinks themselves.
"""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Iterable
from pathlib import Path


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
        if any(candidate.is_relative_to(d) for d in self.deny):
            raise WorkspaceError(f"path is not accessible: {raw}")
        try:
            mode = candidate.stat().st_mode
        except FileNotFoundError:
            return candidate  # write_file is allowed to create it
        if stat.S_ISDIR(mode):
            raise WorkspaceError(f"path is a directory, not a file: {raw}")
        if not stat.S_ISREG(mode):
            raise WorkspaceError(f"path is not a regular file (FIFO, socket or device): {raw}")
        return candidate


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
    atomic rename, so a crash never leaves a truncated file. An existing file's
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
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
