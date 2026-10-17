"""Clone a Forgejo repository to a local directory with ``git``.

``clone_repo`` is the programmatic entry point; ``ForgejoCloneTool`` wraps it as a
:class:`Tool` confined to an agent's workspace. Accepts either a repository ``url``
(``https://host/owner/repo``) or ``instance_url`` + ``repo`` (``owner/name``).

The API token never lands in ``argv`` or in the clone's ``.git/config``: it is passed
to git as an ``Authorization`` header through ``GIT_CONFIG_*`` environment variables.
With a token, redirects are not followed, so it cannot be forwarded to another host.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..workspace import Workspace, WorkspaceError
from .common import ForgejoError, resolve_repo

DEFAULT_CLONE_TIMEOUT = 300.0
_MAX_STDERR = 600


def _git_env(token: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        settings = [
            ("http.extraHeader", f"Authorization: token {token}"),
            ("http.followRedirects", "false"),
        ]
        env["GIT_CONFIG_COUNT"] = str(len(settings))
        for index, (key, value) in enumerate(settings):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


async def _git(
    args: list[str], env: dict[str, str], timeout: float, token: str | None
) -> tuple[int, str, str]:
    """Run ``git`` and return ``(returncode, stdout, stderr)``; the process is killed
    if the call is cancelled or times out."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        raise ForgejoError("git is not installed or not on PATH") from None
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise ForgejoError(f"git {args[0]} timed out after {timeout:g}s") from None
    except BaseException:  # includes cancellation by the registry's timeout
        proc.kill()
        await proc.wait()
        raise
    text_err = err.decode("utf-8", "replace").strip()
    if token:
        text_err = text_err.replace(token, "***")
    return proc.returncode or 0, out.decode("utf-8", "replace").strip(), text_err


def _check_destination(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise ForgejoError(f"destination already exists and is not empty: {dest}")


async def clone_repo(
    *,
    path: str,
    instance_url: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    token: str | None = None,
    branch: str | None = None,
    depth: int | None = None,
    workspace: Workspace | None = None,
    timeout: float = DEFAULT_CLONE_TIMEOUT,
) -> dict[str, Any]:
    """Clone the repo into ``path`` and return ``{"repo", "path", "branch", "commit"}``.

    With a ``workspace``, ``path`` is resolved inside it (relative paths are relative
    to its root) and anything outside, or the root itself, is refused. The destination
    must be missing or an empty directory. Without one, ``path`` is used as given.
    """
    if not isinstance(path, str) or not path.strip():
        raise ForgejoError("path must be a non-empty string")
    base, slug = resolve_repo(instance_url, repo, url)
    if branch is not None and (not branch.strip() or branch.startswith("-")):
        raise ForgejoError(f"invalid branch name: {branch!r}")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise ForgejoError("depth must be a positive integer")
    if workspace is not None:
        try:
            dest = workspace.resolve_directory(path)
        except WorkspaceError as exc:
            raise ForgejoError(str(exc)) from exc
    else:
        dest = Path(path).expanduser().resolve()
    _check_destination(dest)

    owner, name = slug.split("/")
    remote = f"{base}/{quote(owner, safe='')}/{quote(name, safe='')}.git"
    args = ["clone", "--quiet"]
    if branch:
        args += ["--branch", branch]
    if depth:
        args += ["--depth", str(depth)]
    args += ["--", remote, str(dest)]
    existed = dest.exists()
    env = _git_env(token)
    try:
        code, _, err = await _git(args, env, timeout, token)
    except BaseException:
        _discard(dest, existed)
        raise
    if code != 0:
        _discard(dest, existed)
        raise ForgejoError(f"git clone of {base}/{slug} failed: {err[:_MAX_STDERR]}")
    _, commit, _ = await _git(["-C", str(dest), "rev-parse", "HEAD"], env, 30.0, token)
    _, current, _ = await _git(["-C", str(dest), "branch", "--show-current"], env, 30.0, token)
    return {"repo": slug, "path": str(dest), "branch": current or branch, "commit": commit}


def _discard(dest: Path, existed: bool) -> None:
    """Remove what a failed or cancelled clone left behind (git usually cleans up)."""
    with contextlib.suppress(OSError):
        if existed:  # an empty directory we were handed: empty it again, keep it
            for child in dest.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        else:
            shutil.rmtree(dest, ignore_errors=True)


class ForgejoCloneTool:
    """Clone a Forgejo repository into the agent's workspace."""

    name = "forgejo_clone"
    description = (
        "Clone a Forgejo repository into a directory inside the workspace with git and "
        "return its path, branch and HEAD commit (JSON). Provide 'path' (destination, "
        "relative to the workspace root; must not exist or be empty) plus either 'url' "
        "(https://host/owner/repo) or 'instance_url' + 'repo' ('owner/name'). Optional "
        "'token' (API token, for private repos), 'branch' and 'depth' (shallow clone)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination directory, relative to the workspace root.",
            },
            "url": {"type": "string", "description": "Repo URL, e.g. https://host/owner/repo"},
            "instance_url": {
                "type": "string",
                "description": "Instance base URL, e.g. https://forgejo.example.com",
            },
            "repo": {"type": "string", "description": "Repository as 'owner/name'"},
            "token": {"type": "string", "description": "Optional API token"},
            "branch": {"type": "string", "description": "Branch or tag to check out"},
            "depth": {"type": "integer", "description": "Shallow clone depth"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await clone_repo(
            path=arguments.get("path") or "",
            instance_url=arguments.get("instance_url"),
            repo=arguments.get("repo"),
            url=arguments.get("url"),
            token=arguments.get("token"),
            branch=arguments.get("branch"),
            depth=arguments.get("depth"),
            workspace=self._workspace,
        )
        return json.dumps(result, indent=2)
