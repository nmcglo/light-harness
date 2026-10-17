"""Helpers shared by the Forgejo tools: the error type, repo/URL resolution, and the
HTTP call to the API."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlsplit

import httpx


class ForgejoError(Exception):
    """A request that could not be built or that the server rejected."""


def parse_repo_url(url: str) -> tuple[str, str]:
    """Split ``https://host[/prefix]/owner/repo[.git]`` into ``(instance_url, "owner/repo")``."""
    parts = urlsplit(url.strip())
    segments = [s for s in parts.path.split("/") if s]
    if parts.scheme not in ("http", "https") or not parts.netloc or len(segments) < 2:
        raise ForgejoError(f"not a Forgejo repo URL (expected https://host/owner/repo): {url}")
    if parts.username or parts.password:
        raise ForgejoError("repo URL must not embed credentials; pass 'token' instead")
    owner, name = segments[-2], segments[-1].removesuffix(".git")
    prefix = "".join(f"/{s}" for s in segments[:-2])
    return f"{parts.scheme}://{parts.netloc}{prefix}", f"{owner}/{name}"


def resolve_repo(instance_url: str | None, repo: str | None, url: str | None) -> tuple[str, str]:
    """The ``(instance_url, "owner/repo")`` named by a repo ``url`` or by
    ``instance_url`` + ``repo``."""
    if url:
        if instance_url or repo:
            raise ForgejoError("give either url, or instance_url + repo, not both")
        return parse_repo_url(url)
    if not (instance_url and repo):
        raise ForgejoError("give either url, or both instance_url and repo")
    owner, sep, name = repo.strip("/").removesuffix(".git").partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ForgejoError(f"repo must look like 'owner/name', got {repo!r}")
    return instance_url.strip().rstrip("/"), f"{owner}/{name}"


def positive_id(value: Any, label: str) -> int:
    """``value`` as a positive integer, or :class:`ForgejoError`."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ForgejoError(f"{label} must be a positive integer, got {value!r}")
    try:
        number = int(value)
    except ValueError:
        raise ForgejoError(f"{label} must be a positive integer, got {value!r}") from None
    if number < 1:
        raise ForgejoError(f"{label} must be a positive integer, got {value!r}")
    return number


def require_token(token: str | None, what: str) -> str:
    if not token:
        raise ForgejoError(f"{what} requires an API token")
    return token


async def send(
    *,
    base: str,
    slug: str,
    path: str,
    token: str | None,
    client: httpx.AsyncClient | None,
    timeout: float,
    accept: str = "application/json",
    method: str = "GET",
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    ok_status: int | tuple[int, ...] = 200,
) -> httpx.Response:
    """Call ``{base}/api/v1/repos/{owner}/{repo}{path}``; a status outside ``ok_status``
    raises :class:`ForgejoError`."""
    owner, name = slug.split("/")
    endpoint = f"{base}/api/v1/repos/{quote(owner, safe='')}/{quote(name, safe='')}{path}"
    headers = {"Accept": accept}
    if token:
        headers["Authorization"] = f"token {token}"
    own = client is None
    http = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    try:
        try:
            response = await http.request(
                method, endpoint, headers=headers, json=json_body, params=params
            )
        except httpx.HTTPError as exc:
            raise ForgejoError(f"request to {endpoint} failed: {exc}") from exc
    finally:
        if own:
            await http.aclose()
    accepted = (ok_status,) if isinstance(ok_status, int) else ok_status
    if response.status_code not in accepted:
        raise ForgejoError(
            f"{endpoint} returned HTTP {response.status_code}: {response.text[:300]}"
        )
    return response


async def ensure_comment_belongs(
    *,
    base: str,
    slug: str,
    number: int,
    comment_id: int,
    label: str,
    token: str | None,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> None:
    """Raise unless comment ``comment_id`` is on issue/PR ``number``.

    Forgejo's comment endpoints address a comment by id alone, so a wrong id would edit or
    delete a comment elsewhere in the repo. The comment is fetched first and its
    ``issue_url`` / ``pull_request_url`` must end in ``number``; ``label`` (for example
    ``"PR #9"``) names the target in the error."""
    response = await send(
        base=base, slug=slug, path=f"/issues/comments/{comment_id}", token=token,
        client=client, timeout=timeout,
    )  # fmt: skip
    try:
        info = response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc
    owners = [
        str(info.get(key) or "").rstrip("/").rsplit("/", 1)[-1]
        for key in ("pull_request_url", "issue_url")
    ]
    if str(number) not in owners:
        raise ForgejoError(
            f"comment {comment_id} does not belong to {label}; it was left unchanged"
        )
