"""Fetch pull request details from a Forgejo instance via its REST API.

``fetch_pr`` is the programmatic entry point; ``ForgejoPRTool`` wraps it as a
:class:`Tool`. Both accept either ``instance_url`` + ``repo`` + ``pr_number`` or a
full PR ``url`` (``https://host/owner/repo/pulls/N``). The API token comes from the
secret registry (see :class:`.common.ForgejoTool`).
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from ...secrets import SecretValue
from .common import (
    ForgejoError,
    ForgejoTool,
    ensure_comment_belongs,
    positive_id,
    require_token,
    resolve_repo,
    send,
)

_PR_PATH = re.compile(r"^(?P<prefix>.*?)/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pulls/(?P<num>\d+)/?$")


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Split ``https://host[/prefix]/owner/repo/pulls/N`` into
    ``(instance_url, "owner/repo", N)``. Raises :class:`ForgejoError` if malformed."""
    parts = urlsplit(url.strip())
    match = _PR_PATH.match(parts.path)
    if parts.scheme not in ("http", "https") or not parts.netloc or not match:
        raise ForgejoError(
            f"not a Forgejo PR URL (expected https://host/owner/repo/pulls/N): {url}"
        )
    instance = f"{parts.scheme}://{parts.netloc}{match['prefix']}"
    return instance, f"{match['owner']}/{match['repo']}", int(match["num"])


def _resolve(
    instance_url: str | None, repo: str | None, pr_number: int | str | None, url: str | None
) -> tuple[str, str, int]:
    if url:
        if instance_url or repo or pr_number is not None:
            raise ForgejoError("give either url, or instance_url + repo + pr_number, not both")
        return parse_pr_url(url)
    if not (instance_url and repo and pr_number is not None):
        raise ForgejoError("give either url, or all of instance_url, repo and pr_number")
    owner, sep, name = repo.strip("/").partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ForgejoError(f"repo must look like 'owner/name', got {repo!r}")
    try:
        number = int(pr_number)
    except (TypeError, ValueError):
        raise ForgejoError(f"pr_number must be an integer, got {pr_number!r}") from None
    return instance_url.strip().rstrip("/"), f"{owner}/{name}", number


async def _get(
    *,
    instance_url: str | None,
    repo: str | None,
    pr_number: int | str | None,
    url: str | None,
    token: SecretValue | None,
    client: httpx.AsyncClient | None,
    timeout: float,
    suffix: str,
    accept: str,
    method: str = "GET",
    section: str = "pulls",
    json_body: dict[str, Any] | None = None,
    ok_status: int = 200,
) -> httpx.Response:
    """Call a PR endpoint ``/{section}/{n}{suffix}`` for the PR named by ``url`` or
    ``instance_url`` + ``repo`` + ``pr_number``."""
    base, slug, number = _resolve(instance_url, repo, pr_number, url)
    return await send(
        base=base, slug=slug, path=f"/{section}/{number}{suffix}", token=token,
        client=client, timeout=timeout, accept=accept, method=method, json_body=json_body,
        ok_status=ok_status,
    )  # fmt: skip


async def fetch_pr(
    *,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Return the PR's JSON object from ``GET /api/v1/repos/{owner}/{repo}/pulls/{n}``."""
    response = await _get(
        instance_url=instance_url, repo=repo, pr_number=pr_number, url=url, token=token,
        client=client, timeout=timeout, suffix="", accept="application/json",
    )  # fmt: skip
    try:
        data = response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise ForgejoError(f"{response.url} returned unexpected JSON type {type(data).__name__}")
    return data


async def fetch_pr_diff(
    *,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> str:
    """Return the PR's unified diff text from ``GET .../pulls/{n}.diff``."""
    response = await _get(
        instance_url=instance_url, repo=repo, pr_number=pr_number, url=url, token=token,
        client=client, timeout=timeout, suffix=".diff", accept="text/plain",
    )  # fmt: skip
    return response.text


_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Full PR URL, e.g. https://host/owner/repo/pulls/9"},
        "instance_url": {
            "type": "string",
            "description": "Instance base URL, e.g. https://forgejo.example.com",
        },
        "repo": {"type": "string", "description": "Repository as 'owner/name'"},
        "pr_number": {"type": "integer"},
    },
}
_ARG_KEYS = ("instance_url", "repo", "pr_number", "url")


class ForgejoPRTool(ForgejoTool):
    """Fetch a Forgejo pull request's details as JSON."""

    name = "forgejo_pr"
    untrusted_output = True
    description = (
        "Fetch a Forgejo pull request's details (JSON) via the Forgejo API. Provide either "
        "'url' (full PR URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = _PARAMETERS

    async def run(self, arguments: dict[str, Any]) -> str:
        pr = await fetch_pr(**self._args(arguments, _ARG_KEYS))
        return json.dumps(pr, indent=2)


class ForgejoPRDiffTool(ForgejoTool):
    """Fetch a Forgejo pull request's unified diff as text."""

    name = "forgejo_pr_diff"
    untrusted_output = True
    description = (
        "Fetch a Forgejo pull request's unified diff text via the Forgejo API. Provide either "
        "'url' (full PR URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = _PARAMETERS

    async def run(self, arguments: dict[str, Any]) -> str:
        return await fetch_pr_diff(**self._args(arguments, _ARG_KEYS))


async def post_pr_comment(
    *,
    body: str,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Post a comment on the PR via ``POST .../issues/{n}/comments``; returns the
    created comment's JSON. Forgejo requires a token with write access to issues."""
    if not isinstance(body, str) or not body.strip():
        raise ForgejoError("comment body must be a non-empty string")
    token = require_token(token, "posting a comment")
    response = await _get(
        instance_url=instance_url, repo=repo, pr_number=pr_number, url=url, token=token,
        client=client, timeout=timeout, suffix="/comments", accept="application/json",
        method="POST", section="issues", json_body={"body": body}, ok_status=201,
    )  # fmt: skip
    try:
        data = response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise ForgejoError(f"{response.url} returned unexpected JSON type {type(data).__name__}")
    return data


class ForgejoPRCommentTool(ForgejoTool):
    """Post a comment on a Forgejo pull request."""

    name = "forgejo_pr_comment"
    description = (
        "Post a comment on a Forgejo pull request via the Forgejo API and return the created "
        "comment (JSON). Provide 'body' plus either 'url' (full PR URL) or 'instance_url' + "
        "'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "body": {"type": "string", "description": "Comment text (Markdown)"},
        },
        "required": ["body"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        comment = await post_pr_comment(
            body=arguments.get("body") or "", **self._args(arguments, _ARG_KEYS)
        )
        return json.dumps(comment, indent=2)


REVIEW_EVENTS = ("APPROVED", "REQUEST_CHANGES", "COMMENT")
_EVENT_ALIASES = {"APPROVE": "APPROVED", "REQUEST_CHANGE": "REQUEST_CHANGES"}


def _review_comment(entry: Any, index: int) -> dict[str, Any]:
    """Validate one inline comment and map it to Forgejo's ``CreatePullReviewComment``."""
    where = f"comments[{index}]"
    if not isinstance(entry, dict):
        raise ForgejoError(f"{where} must be an object with path, body and new_line/old_line")
    path, text = entry.get("path"), entry.get("body")
    if not isinstance(path, str) or not path:
        raise ForgejoError(f"{where}.path must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ForgejoError(f"{where}.body must be a non-empty string")
    out: dict[str, Any] = {"path": path, "body": text}
    for key, field in (("new_line", "new_position"), ("old_line", "old_position")):
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ForgejoError(f"{where}.{key} must be a positive integer")
        out[field] = value
    if len(out) == 2:
        raise ForgejoError(
            f"{where} needs new_line (added/context line) or old_line (removed line)"
        )
    return out


async def submit_pr_review(
    *,
    event: str,
    body: str = "",
    comments: list[dict[str, Any]] | None = None,
    commit_id: str | None = None,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Submit a review via ``POST .../pulls/{n}/reviews``; returns the review's JSON.

    ``event`` is ``APPROVED``, ``REQUEST_CHANGES`` or ``COMMENT``. ``comments`` are inline
    comments: ``{"path", "body", "new_line"?, "old_line"?}`` where ``new_line`` is a line
    number in the new version of the file and ``old_line`` one in the old version (for
    removed lines). ``commit_id`` pins the review to a head commit SHA.
    """
    normalized = str(event).strip().upper()
    normalized = _EVENT_ALIASES.get(normalized, normalized)
    if normalized not in REVIEW_EVENTS:
        raise ForgejoError(f"event must be one of {', '.join(REVIEW_EVENTS)}, got {event!r}")
    if not isinstance(body, str):
        raise ForgejoError("review body must be a string")
    if comments is not None and not isinstance(comments, list):
        raise ForgejoError("comments must be a list")
    inline = [_review_comment(c, i) for i, c in enumerate(comments or [])]
    if normalized != "APPROVED" and not body.strip() and not inline:
        raise ForgejoError(f"a {normalized} review needs a body or at least one inline comment")
    token = require_token(token, "submitting a review")
    payload: dict[str, Any] = {"event": normalized, "body": body}
    if inline:
        payload["comments"] = inline
    if commit_id:
        payload["commit_id"] = commit_id
    response = await _get(
        instance_url=instance_url, repo=repo, pr_number=pr_number, url=url, token=token,
        client=client, timeout=timeout, suffix="/reviews", accept="application/json",
        method="POST", json_body=payload, ok_status=200,
    )  # fmt: skip
    try:
        data = response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise ForgejoError(f"{response.url} returned unexpected JSON type {type(data).__name__}")
    return data


class ForgejoPRReviewTool(ForgejoTool):
    """Submit a review (approve / request changes / comment) on a Forgejo pull request."""

    name = "forgejo_pr_review"
    description = (
        "Submit a review on a Forgejo pull request and return it (JSON). 'event' is APPROVED, "
        "REQUEST_CHANGES or COMMENT. 'body' is the overall review text. 'comments' are optional "
        "inline comments on specific lines: each has 'path', 'body' and 'new_line' (line number "
        "in the new file, for added/context lines) and/or 'old_line' (line number in the old "
        "file, for removed lines). Optional 'commit_id' pins the head commit SHA. Provide either "
        "'url' (full PR URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "event": {"type": "string", "enum": list(REVIEW_EVENTS)},
            "body": {"type": "string", "description": "Overall review text (Markdown)"},
            "commit_id": {"type": "string", "description": "Optional head commit SHA"},
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path in the repo"},
                        "body": {"type": "string"},
                        "new_line": {"type": "integer", "description": "Line in the new file"},
                        "old_line": {"type": "integer", "description": "Line in the old file"},
                    },
                    "required": ["path", "body"],
                },
            },
        },
        "required": ["event"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        review = await submit_pr_review(
            event=arguments.get("event") or "",
            body=arguments.get("body") or "",
            comments=arguments.get("comments"),
            commit_id=arguments.get("commit_id"),
            **self._args(arguments, _ARG_KEYS),
        )
        return json.dumps(review, indent=2)


async def delete_pr_comment(
    *,
    comment_id: int | str,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Delete comment ``comment_id`` from the PR via ``DELETE .../issues/comments/{id}``.

    The comment is fetched first and must belong to the named PR, so a wrong id (or PR)
    cannot delete a comment elsewhere in the repo. Returns ``{"deleted": True, ...}``.
    """
    cid = positive_id(comment_id, "comment_id")
    token = require_token(token, "deleting a comment")
    base, slug, number = _resolve(instance_url, repo, pr_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    await ensure_comment_belongs(**common, number=number, comment_id=cid, label=f"PR #{number}")
    await send(**common, path=f"/issues/comments/{cid}", method="DELETE", ok_status=204)
    return {"deleted": True, "comment_id": cid, "pr_number": number}


async def delete_pr_review(
    *,
    review_id: int | str,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Delete review ``review_id`` from the PR via ``DELETE .../pulls/{n}/reviews/{id}``
    (the review's own inline comments go with it). Returns ``{"deleted": True, ...}``."""
    rid = positive_id(review_id, "review_id")
    token = require_token(token, "deleting a review")
    _, _, number = _resolve(instance_url, repo, pr_number, url)
    await _get(
        instance_url=instance_url, repo=repo, pr_number=pr_number, url=url, token=token,
        client=client, timeout=timeout, suffix=f"/reviews/{rid}", accept="application/json",
        method="DELETE", ok_status=204,
    )  # fmt: skip
    return {"deleted": True, "review_id": rid, "pr_number": number}


class ForgejoPRCommentDeleteTool(ForgejoTool):
    """Delete a comment from a Forgejo pull request."""

    name = "forgejo_pr_comment_delete"
    description = (
        "Delete a comment from a Forgejo pull request via the Forgejo API (irreversible). "
        "'comment_id' is the comment's numeric id (the 'id' returned by forgejo_pr_comment, "
        "or the number in '#issuecomment-N' of its link); it must belong to the named PR. "
        "Provide either 'url' (full PR URL) or 'instance_url' + 'repo' ('owner/name') + "
        "'pr_number'."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "comment_id": {"type": "integer", "description": "Numeric comment id"},
        },
        "required": ["comment_id"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await delete_pr_comment(
            comment_id=arguments.get("comment_id"), **self._args(arguments, _ARG_KEYS)
        )
        return json.dumps(result)


class ForgejoPRReviewDeleteTool(ForgejoTool):
    """Delete a review from a Forgejo pull request."""

    name = "forgejo_pr_review_delete"
    description = (
        "Delete a review from a Forgejo pull request via the Forgejo API (irreversible; "
        "its inline comments are removed too). 'review_id' is the review's numeric id "
        "(the 'id' returned by forgejo_pr_review). Provide either 'url' (full PR URL) or "
        "'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "review_id": {"type": "integer", "description": "Numeric review id"},
        },
        "required": ["review_id"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await delete_pr_review(
            review_id=arguments.get("review_id"), **self._args(arguments, _ARG_KEYS)
        )
        return json.dumps(result)


async def edit_pr_comment(
    *,
    comment_id: int | str,
    body: str,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Replace the text of comment ``comment_id`` on the PR (``PATCH .../issues/comments/{id}``)
    and return the updated comment. The comment must belong to the named PR."""
    cid = positive_id(comment_id, "comment_id")
    if not isinstance(body, str) or not body.strip():
        raise ForgejoError("comment body must be a non-empty string")
    token = require_token(token, "editing a comment")
    base, slug, number = _resolve(instance_url, repo, pr_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    await ensure_comment_belongs(**common, number=number, comment_id=cid, label=f"PR #{number}")
    response = await send(
        **common, path=f"/issues/comments/{cid}", method="PATCH", json_body={"body": body}
    )
    data = response.json()
    if not isinstance(data, dict):
        raise ForgejoError(f"{response.url} returned unexpected JSON type {type(data).__name__}")
    return data


class ForgejoPRCommentEditTool(ForgejoTool):
    """Edit a comment on a Forgejo pull request."""

    name = "forgejo_pr_comment_edit"
    description = (
        "Replace the text of a comment on a Forgejo pull request and return the updated "
        "comment (JSON). 'comment_id' is the comment's numeric id (the 'id' returned by "
        "forgejo_pr_comment, or the number in '#issuecomment-N' of its link); it must belong "
        "to the named PR. 'body' is the new text (Markdown). Provide either 'url' (full PR "
        "URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'. The "
        "configured token must be allowed to edit that comment."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "comment_id": {"type": "integer", "description": "Numeric comment id"},
            "body": {"type": "string", "description": "New comment text (Markdown)"},
        },
        "required": ["comment_id", "body"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        comment = await edit_pr_comment(
            comment_id=arguments.get("comment_id"),
            body=arguments.get("body") or "",
            **self._args(arguments, _ARG_KEYS),
        )
        return json.dumps(comment, indent=2)


MERGE_METHODS = ("merge", "rebase", "rebase-merge", "squash", "fast-forward-only")


def _dict_of(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise ForgejoError(f"{response.url} returned unexpected JSON type {type(data).__name__}")
    return data


async def create_pr(
    *,
    title: str,
    head: str,
    base: str | None = None,
    body: str = "",
    instance_url: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Open a PR merging branch ``head`` into ``base`` (the repo's default branch when
    omitted) via ``POST .../pulls``; returns the new PR. ``head`` may be ``user:branch``
    for a branch in a fork. Labels and assignees can be set afterwards with
    ``forgejo_issue_update``, which works on PRs too."""
    if not isinstance(title, str) or not title.strip():
        raise ForgejoError("title must be a non-empty string")
    if not isinstance(head, str) or not head.strip():
        raise ForgejoError("head (the branch to merge) must be a non-empty string")
    if not isinstance(body, str):
        raise ForgejoError("body must be a string")
    token = require_token(token, "creating a PR")
    instance, slug = resolve_repo(instance_url, repo, url)
    common: dict[str, Any] = {
        "base": instance, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    target = base.strip() if isinstance(base, str) and base.strip() else None
    if target is None:
        target = _dict_of(await send(**common, path=""))["default_branch"]
    response = await send(
        **common, path="/pulls", method="POST", ok_status=201,
        json_body={"title": title.strip(), "head": head.strip(), "base": target, "body": body},
    )  # fmt: skip
    return _dict_of(response)


async def close_pr(
    *,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Close the PR without merging it (``PATCH .../pulls/{n}`` with ``state: closed``);
    returns the PR as it now is."""
    token = require_token(token, "closing a PR")
    instance, slug, number = _resolve(instance_url, repo, pr_number, url)
    response = await send(
        base=instance, slug=slug, path=f"/pulls/{number}", token=token, client=client,
        timeout=timeout, method="PATCH", json_body={"state": "closed"}, ok_status=(200, 201),
    )  # fmt: skip
    return _dict_of(response)


async def merge_pr(
    *,
    method: str = "merge",
    title: str | None = None,
    message: str | None = None,
    delete_branch: bool = False,
    head_commit_id: str | None = None,
    instance_url: str | None = None,
    repo: str | None = None,
    pr_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Merge the PR (``POST .../pulls/{n}/merge``) and return what the server now says:
    ``{"merged", "state", "merge_commit_sha", "pr_number", "html_url"}``.

    ``method`` is one of ``MERGE_METHODS``; ``title`` / ``message`` set the merge commit's
    subject and body; ``delete_branch`` removes the head branch afterwards;
    ``head_commit_id`` makes the merge fail unless the PR head is still that commit."""
    if method not in MERGE_METHODS:
        raise ForgejoError(f"method must be one of {', '.join(MERGE_METHODS)}, got {method!r}")
    token = require_token(token, "merging a PR")
    instance, slug, number = _resolve(instance_url, repo, pr_number, url)
    payload: dict[str, Any] = {"Do": method}
    if title:
        payload["MergeTitleField"] = title
    if message:
        payload["MergeMessageField"] = message
    if delete_branch:
        payload["delete_branch_after_merge"] = True
    if head_commit_id:
        payload["head_commit_id"] = head_commit_id
    common: dict[str, Any] = {
        "base": instance, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    await send(**common, path=f"/pulls/{number}/merge", method="POST", json_body=payload)
    pr = _dict_of(await send(**common, path=f"/pulls/{number}"))
    return {
        "merged": pr.get("merged"),
        "state": pr.get("state"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "pr_number": number,
        "html_url": pr.get("html_url"),
    }


class ForgejoPRCreateTool(ForgejoTool):
    """Open a pull request in a Forgejo repository."""

    name = "forgejo_pr_create"
    description = (
        "Open a pull request in a Forgejo repository and return it (JSON). 'head' is the "
        "branch to merge ('user:branch' for a branch in a fork), 'base' the branch to merge "
        "into (the repo's default branch if omitted); 'title' is required, 'body' (Markdown) "
        "optional. Labels and assignees can be set afterwards with forgejo_issue_update. "
        "Provide either 'url' (repo URL, https://host/owner/repo) or 'instance_url' + 'repo' "
        "('owner/name')."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Repo URL, e.g. https://host/owner/repo"},
            "instance_url": _PARAMETERS["properties"]["instance_url"],
            "repo": _PARAMETERS["properties"]["repo"],
            "title": {"type": "string"},
            "head": {"type": "string", "description": "Branch to merge, or 'user:branch'"},
            "base": {"type": "string", "description": "Target branch (default: repo default)"},
            "body": {"type": "string", "description": "PR description (Markdown)"},
        },
        "required": ["title", "head"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        pr = await create_pr(
            title=arguments.get("title") or "",
            head=arguments.get("head") or "",
            base=arguments.get("base"),
            body=arguments.get("body") or "",
            **self._args(arguments, ("instance_url", "repo", "url")),
        )
        return json.dumps(pr, indent=2)


class ForgejoPRCloseTool(ForgejoTool):
    """Close a Forgejo pull request without merging it."""

    name = "forgejo_pr_close"
    description = (
        "Close a Forgejo pull request WITHOUT merging it and return the PR (JSON). Provide "
        "either 'url' (full PR URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = _PARAMETERS

    async def run(self, arguments: dict[str, Any]) -> str:
        pr = await close_pr(**self._args(arguments, _ARG_KEYS))
        return json.dumps(pr, indent=2)


class ForgejoPRMergeTool(ForgejoTool):
    """Merge a Forgejo pull request."""

    name = "forgejo_pr_merge"
    description = (
        "Merge a Forgejo pull request into its base branch (irreversible). 'method' is merge "
        "(default), rebase, rebase-merge, squash or fast-forward-only; 'title' / 'message' "
        "set the merge commit's subject / body; 'delete_branch' removes the head branch "
        "afterwards; 'head_commit_id' makes the merge fail unless the PR head is still that "
        "commit. Fails if the PR is not mergeable (conflicts, failing checks, required "
        "approvals). Returns merged, state and merge_commit_sha. Provide either 'url' (full "
        "PR URL) or 'instance_url' + 'repo' ('owner/name') + 'pr_number'."
    )
    parameters: dict[str, Any] = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "method": {"type": "string", "enum": list(MERGE_METHODS)},
            "title": {"type": "string", "description": "Merge commit title"},
            "message": {"type": "string", "description": "Merge commit message"},
            "delete_branch": {"type": "boolean", "description": "Delete the head branch after"},
            "head_commit_id": {"type": "string", "description": "Required head commit SHA"},
        },
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await merge_pr(
            method=arguments.get("method") or "merge",
            title=arguments.get("title"),
            message=arguments.get("message"),
            delete_branch=bool(arguments.get("delete_branch")),
            head_commit_id=arguments.get("head_commit_id"),
            **self._args(arguments, _ARG_KEYS),
        )
        return json.dumps(result, indent=2)
