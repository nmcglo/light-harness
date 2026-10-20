"""forgejo_clone: confinement to the workspace, token handling, cleanup on failure.

A throwaway local HTTP server serves a bare repository over git's dumb protocol, so
the real ``git clone`` runs without any network access.
"""

from __future__ import annotations

import asyncio
import functools
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from light_harness.secrets import (
    SecretContext,
    SecretPolicy,
    SecretRegistry,
    SecretSpec,
    ToolSecrets,
)
from light_harness.tools import build_tool_registry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

TOKEN = "s3cr3t-token"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


class _Handler(SimpleHTTPRequestHandler):
    seen_auth: list[str | None] = []

    def do_GET(self) -> None:
        type(self).seen_auth.append(self.headers.get("Authorization"))
        super().do_GET()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def forge(tmp_path: Path) -> Iterator[str]:
    """Base URL of a server hosting ``owner/repo`` (one commit, file ``hello.txt``)."""
    work = tmp_path / "work"
    work.mkdir()
    _git("init", "-q", "-b", "main", cwd=work)
    (work / "hello.txt").write_text("hi\n")
    _git("add", ".", cwd=work)
    _git("commit", "-q", "-m", "init", cwd=work)
    served = tmp_path / "served"
    (served / "owner").mkdir(parents=True)
    _git("clone", "-q", "--bare", str(work), str(served / "owner" / "repo.git"), cwd=tmp_path)
    _git("update-server-info", cwd=served / "owner" / "repo.git")

    _Handler.seen_auth = []
    handler = functools.partial(_Handler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(home))
    monkeypatch.setenv("CLONE_TEST_TOKEN", TOKEN)
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _clone(workspace: Path, forge: str, **args: object) -> str:
    """Run the tool with the token found in the registry by the forge's host."""
    secrets = SecretRegistry(workspace.parent / "home", audit=lambda _line: None)
    secrets.define(
        "forge",
        SecretSpec(
            source="env:CLONE_TEST_TOKEN",
            allow=SecretPolicy(tools=["forgejo_clone"], hosts=["127.0.0.1"]),
        ),
    )
    context = SecretContext(agent="a", interaction="chat")
    registry = build_tool_registry(["forgejo_clone"], workspace, ToolSecrets(secrets, context))
    arguments = {"instance_url": forge, "repo": "owner/repo", **args}
    return asyncio.run(registry.run("forgejo_clone", arguments))


def test_clone_lands_in_workspace_and_keeps_the_token_out_of_the_repo(
    ws: Path, forge: str
) -> None:
    result = _clone(ws, forge, path="repos/mine")

    assert "Error" not in result
    assert (ws / "repos" / "mine" / "hello.txt").read_text() == "hi\n"
    assert TOKEN not in (ws / "repos" / "mine" / ".git" / "config").read_text()
    # The token reached the server as an Authorization header (and nowhere else).
    assert f"token {TOKEN}" in _Handler.seen_auth


@pytest.mark.parametrize("target", ["../escape", "/tmp/forgejo-clone-escape", "."])
def test_paths_outside_the_workspace_or_the_root_itself_are_refused(
    ws: Path, forge: str, target: str
) -> None:
    result = _clone(ws, forge, path=target)

    assert result.startswith("Error")
    assert list(ws.iterdir()) == []
    assert not Path("/tmp/forgejo-clone-escape").exists()


def test_symlink_pointing_out_of_the_workspace_is_refused(
    ws: Path, forge: str, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (ws / "link").symlink_to(outside)

    assert _clone(ws, forge, path="link/sub").startswith("Error")
    assert list(outside.iterdir()) == []


def test_non_empty_destination_is_refused_and_left_untouched(ws: Path, forge: str) -> None:
    (ws / "taken").mkdir()
    (ws / "taken" / "keep.txt").write_text("mine\n")

    assert "not empty" in _clone(ws, forge, path="taken")
    assert (ws / "taken" / "keep.txt").read_text() == "mine\n"


def test_failed_clone_leaves_nothing_behind(ws: Path, forge: str) -> None:
    result = _clone(ws, forge, repo="owner/missing", path="gone")

    assert result.startswith("Error")
    assert not (ws / "gone").exists()


def test_tool_needs_a_workspace() -> None:
    assert "forgejo_clone" not in build_tool_registry(None).names()
