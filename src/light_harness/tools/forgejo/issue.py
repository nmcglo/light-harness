"""Forgejo issues: create, read, list, update (title, body, state, assignees, labels)
and comment, through the Forgejo REST API.

Repo-level calls (``create_issue``, ``list_issues``) take a repo ``url``
(``https://host/owner/repo``) or ``instance_url`` + ``repo`` (``owner/name``). Calls on
one issue take its ``url`` (``https://host/owner/repo/issues/N``) or ``instance_url`` +
``repo`` + ``issue_number``. Labels are given by name and resolved to ids here.
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

_ISSUE_PATH = re.compile(
    r"^(?P<prefix>.*?)/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?:issues|pulls)/(?P<num>\d+)/?$"
)
ISSUE_STATES = ("open", "closed")
MAX_PAGE_SIZE = 50  # Forgejo's cap on items per page


def parse_issue_url(url: str) -> tuple[str, str, int]:
    """Split ``https://host[/prefix]/owner/repo/issues/N`` into
    ``(instance_url, "owner/repo", N)``."""
    parts = urlsplit(url.strip())
    match = _ISSUE_PATH.match(parts.path)
    if parts.scheme not in ("http", "https") or not parts.netloc or not match:
        raise ForgejoError(
            f"not a Forgejo issue URL (expected https://host/owner/repo/issues/N): {url}"
        )
    instance = f"{parts.scheme}://{parts.netloc}{match['prefix']}"
    return instance, f"{match['owner']}/{match['repo']}", int(match["num"])


def _resolve_issue(
    instance_url: str | None, repo: str | None, issue_number: int | str | None, url: str | None
) -> tuple[str, str, int]:
    if url:
        if instance_url or repo or issue_number is not None:
            raise ForgejoError("give either url, or instance_url + repo + issue_number, not both")
        return parse_issue_url(url)
    if issue_number is None:
        raise ForgejoError("give either url, or all of instance_url, repo and issue_number")
    base, slug = resolve_repo(instance_url, repo, None)
    return base, slug, positive_id(issue_number, "issue_number")


def _names(value: Any, label: str) -> list[str]:
    """``value`` as a list of non-empty strings (``None`` -> ``[]``)."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ForgejoError(f"{label} must be a list of non-empty strings")
    return [v.strip() for v in value]


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ForgejoError(f"{response.url} did not return JSON") from exc


async def _label_ids(
    names: list[str], *, base: str, slug: str, token: SecretValue | None,
    client: httpx.AsyncClient | None, timeout: float,
) -> list[int]:  # fmt: skip
    """The ids of the repo labels called ``names``; an unknown name is an error that
    lists the labels that do exist."""
    if not names:
        return []
    known: dict[str, int] = {}
    for page in range(1, 21):
        response = await send(
            base=base, slug=slug, path="/labels", token=token, client=client, timeout=timeout,
            params={"limit": MAX_PAGE_SIZE, "page": page},
        )  # fmt: skip
        items = _json(response)
        if not isinstance(items, list):
            raise ForgejoError(f"{response.url} returned unexpected JSON")
        known.update({str(item["name"]): int(item["id"]) for item in items})
        if len(items) < MAX_PAGE_SIZE:
            break
    missing = [n for n in names if n not in known]
    if missing:
        raise ForgejoError(
            f"unknown label(s): {', '.join(missing)}; the repo has: "
            f"{', '.join(sorted(known)) or '(no labels)'}"
        )
    return [known[n] for n in names]


def _summary(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "state": issue.get("state"),
        "user": (issue.get("user") or {}).get("login"),
        "labels": [label.get("name") for label in issue.get("labels") or []],
        "assignees": [a.get("login") for a in issue.get("assignees") or []],
        "comments": issue.get("comments"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "html_url": issue.get("html_url"),
    }


async def create_issue(
    *,
    title: str,
    body: str = "",
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    instance_url: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Open a new issue and return it (``POST .../issues``)."""
    if not isinstance(title, str) or not title.strip():
        raise ForgejoError("title must be a non-empty string")
    if not isinstance(body, str):
        raise ForgejoError("body must be a string")
    token = require_token(token, "creating an issue")
    label_names = _names(labels, "labels")
    base, slug = resolve_repo(instance_url, repo, url)
    payload: dict[str, Any] = {"title": title.strip(), "body": body}
    if label_names:
        payload["labels"] = await _label_ids(
            label_names, base=base, slug=slug, token=token, client=client, timeout=timeout
        )
    if assignees is not None:
        payload["assignees"] = _names(assignees, "assignees")
    response = await send(
        base=base, slug=slug, path="/issues", token=token, client=client, timeout=timeout,
        method="POST", json_body=payload, ok_status=201,
    )  # fmt: skip
    return _json(response)


async def list_issues(
    *,
    state: str = "open",
    labels: list[str] | None = None,
    query: str | None = None,
    limit: int = 30,
    page: int = 1,
    instance_url: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """List issues (not PRs), newest first, as short summaries.

    ``state`` is ``open``, ``closed`` or ``all``; ``labels`` keeps issues carrying all of
    the named labels; ``query`` is a text search. Returns ``{"issues": [...], "page",
    "limit", "total"}``; ``total`` is the server's count across all pages when it reports
    one."""
    if state not in (*ISSUE_STATES, "all"):
        raise ForgejoError(f"state must be open, closed or all, got {state!r}")
    size = positive_id(limit, "limit")
    if size > MAX_PAGE_SIZE:
        raise ForgejoError(f"limit must be at most {MAX_PAGE_SIZE}")
    number = positive_id(page, "page")
    base, slug = resolve_repo(instance_url, repo, url)
    params: dict[str, Any] = {"state": state, "type": "issues", "limit": size, "page": number}
    label_names = _names(labels, "labels")
    if label_names:
        params["labels"] = ",".join(label_names)
    if query:
        params["q"] = query
    response = await send(
        base=base, slug=slug, path="/issues", token=token, client=client, timeout=timeout,
        params=params,
    )  # fmt: skip
    items = _json(response)
    if not isinstance(items, list):
        raise ForgejoError(f"{response.url} returned unexpected JSON")
    total = response.headers.get("X-Total-Count")
    return {
        "issues": [_summary(item) for item in items],
        "page": number,
        "limit": size,
        "total": int(total) if total and total.isdigit() else None,
    }


async def get_issue(
    *,
    include_comments: bool = False,
    instance_url: str | None = None,
    repo: str | None = None,
    issue_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """The issue's JSON (``GET .../issues/{n}``); with ``include_comments`` it also gets a
    ``comment_list`` key holding the issue's comments."""
    base, slug, number = _resolve_issue(instance_url, repo, issue_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    issue = _json(await send(**common, path=f"/issues/{number}"))
    if not isinstance(issue, dict):
        raise ForgejoError(f"issue #{number}: unexpected JSON type {type(issue).__name__}")
    if include_comments:
        issue["comment_list"] = _json(await send(**common, path=f"/issues/{number}/comments"))
    return issue


async def update_issue(
    *,
    title: str | None = None,
    body: str | None = None,
    state: str | None = None,
    assignees: list[str] | None = None,
    labels: list[str] | None = None,
    instance_url: str | None = None,
    repo: str | None = None,
    issue_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Change an issue and return it as it now is.

    Only the fields you pass change. ``state`` is ``open`` or ``closed`` (close / reopen);
    ``assignees`` and ``labels`` *replace* the current ones (``[]`` clears them)."""
    if all(v is None for v in (title, body, state, assignees, labels)):
        raise ForgejoError("nothing to update: pass title, body, state, assignees or labels")
    if state is not None and state not in ISSUE_STATES:
        raise ForgejoError(f"state must be open or closed, got {state!r}")
    if title is not None and not title.strip():
        raise ForgejoError("title must not be empty")
    token = require_token(token, "updating an issue")
    base, slug, number = _resolve_issue(instance_url, repo, issue_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    label_ids = None
    if labels is not None:
        label_ids = await _label_ids(_names(labels, "labels"), **common)
    patch: dict[str, Any] = {}
    if title is not None:
        patch["title"] = title.strip()
    if body is not None:
        patch["body"] = body
    if state is not None:
        patch["state"] = state
    if assignees is not None:
        patch["assignees"] = _names(assignees, "assignees")
    if patch:
        await send(
            **common, path=f"/issues/{number}", method="PATCH", json_body=patch,
            ok_status=(200, 201),
        )  # fmt: skip
    if label_ids is not None:
        await send(
            **common, path=f"/issues/{number}/labels", method="PUT",
            json_body={"labels": label_ids},
        )  # fmt: skip
    issue = _json(await send(**common, path=f"/issues/{number}"))
    if not isinstance(issue, dict):
        raise ForgejoError(f"issue #{number}: unexpected JSON type {type(issue).__name__}")
    return issue


async def comment_on_issue(
    *,
    body: str,
    instance_url: str | None = None,
    repo: str | None = None,
    issue_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Post a comment on the issue (``POST .../issues/{n}/comments``); returns it."""
    if not isinstance(body, str) or not body.strip():
        raise ForgejoError("comment body must be a non-empty string")
    token = require_token(token, "commenting on an issue")
    base, slug, number = _resolve_issue(instance_url, repo, issue_number, url)
    response = await send(
        base=base, slug=slug, path=f"/issues/{number}/comments", token=token, client=client,
        timeout=timeout, method="POST", json_body={"body": body}, ok_status=201,
    )  # fmt: skip
    return _json(response)


_REPO_PROPERTIES: dict[str, Any] = {
    "url": {"type": "string", "description": "Repo URL, e.g. https://host/owner/repo"},
    "instance_url": {
        "type": "string",
        "description": "Instance base URL, e.g. https://forgejo.example.com",
    },
    "repo": {"type": "string", "description": "Repository as 'owner/name'"},
}
_ISSUE_PROPERTIES: dict[str, Any] = {
    **_REPO_PROPERTIES,
    "url": {
        "type": "string",
        "description": "Issue URL, e.g. https://host/owner/repo/issues/3",
    },
    "issue_number": {"type": "integer"},
}
_REPO_KEYS = ("instance_url", "repo", "url")
_ISSUE_KEYS = ("instance_url", "repo", "issue_number", "url")
_WHERE_REPO = "Provide either 'url' (repo URL) or 'instance_url' + 'repo' ('owner/name')."
_WHERE_ISSUE = (
    "Provide either 'url' (issue URL) or 'instance_url' + 'repo' ('owner/name') + "
    "'issue_number'."
)
_LABELS = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Label names (must already exist in the repo)",
}
_ASSIGNEES = {"type": "array", "items": {"type": "string"}, "description": "Usernames"}


class ForgejoIssueCreateTool(ForgejoTool):
    """Open an issue in a Forgejo repository."""

    name = "forgejo_issue_create"
    description = (
        "Create an issue in a Forgejo repository and return it (JSON). 'title' is required; "
        "'body' (Markdown), 'labels' (existing label names) and 'assignees' (usernames) are "
        f"optional. {_WHERE_REPO}"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_REPO_PROPERTIES,
            "title": {"type": "string"},
            "body": {"type": "string", "description": "Issue text (Markdown)"},
            "labels": _LABELS,
            "assignees": _ASSIGNEES,
        },
        "required": ["title"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        issue = await create_issue(
            title=arguments.get("title") or "",
            body=arguments.get("body") or "",
            labels=arguments.get("labels"),
            assignees=arguments.get("assignees"),
            **self._args(arguments, _REPO_KEYS),
        )
        return json.dumps(issue, indent=2)


class ForgejoIssueListTool(ForgejoTool):
    """List a Forgejo repository's issues."""

    name = "forgejo_issue_list"
    untrusted_output = True
    description = (
        "List issues (not pull requests) of a Forgejo repository as short summaries (number, "
        "title, state, author, labels, assignees, comment count, dates, link), newest first. "
        "'state' is open (default), closed or all; 'labels' keeps issues with all the named "
        "labels; 'query' searches text; 'limit' (default 30, max 50) and 'page' paginate "
        f"(the reply includes 'total'). {_WHERE_REPO}"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_REPO_PROPERTIES,
            "state": {"type": "string", "enum": [*ISSUE_STATES, "all"]},
            "labels": _LABELS,
            "query": {"type": "string", "description": "Text to search for"},
            "limit": {"type": "integer", "description": "Items per page (max 50)"},
            "page": {"type": "integer", "description": "Page number, from 1"},
        },
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await list_issues(
            state=arguments.get("state") or "open",
            labels=arguments.get("labels"),
            query=arguments.get("query"),
            limit=arguments.get("limit") or 30,
            page=arguments.get("page") or 1,
            **self._args(arguments, _REPO_KEYS),
        )
        return json.dumps(result, indent=2)


class ForgejoIssueTool(ForgejoTool):
    """Read one Forgejo issue."""

    name = "forgejo_issue"
    untrusted_output = True
    description = (
        "Fetch one Forgejo issue's details (JSON): title, body, state, labels, assignees, "
        "dates. Set 'include_comments' to also get its comments as 'comment_list'. "
        f"{_WHERE_ISSUE}"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_ISSUE_PROPERTIES,
            "include_comments": {"type": "boolean", "description": "Also fetch the comments"},
        },
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        issue = await get_issue(
            include_comments=bool(arguments.get("include_comments")),
            **self._args(arguments, _ISSUE_KEYS),
        )
        return json.dumps(issue, indent=2)


class ForgejoIssueUpdateTool(ForgejoTool):
    """Edit, close or reopen a Forgejo issue."""

    name = "forgejo_issue_update"
    description = (
        "Change a Forgejo issue and return it as it now is (JSON). Pass only what should "
        "change: 'title', 'body', 'state' (open / closed: reopen or close the issue), "
        "'assignees' and 'labels' (both REPLACE the current ones; [] clears them). "
        f"{_WHERE_ISSUE}"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_ISSUE_PROPERTIES,
            "title": {"type": "string"},
            "body": {"type": "string", "description": "New issue text (Markdown)"},
            "state": {"type": "string", "enum": list(ISSUE_STATES)},
            "labels": _LABELS,
            "assignees": _ASSIGNEES,
        },
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        issue = await update_issue(
            title=arguments.get("title"),
            body=arguments.get("body"),
            state=arguments.get("state"),
            assignees=arguments.get("assignees"),
            labels=arguments.get("labels"),
            **self._args(arguments, _ISSUE_KEYS),
        )
        return json.dumps(issue, indent=2)


class ForgejoIssueCommentTool(ForgejoTool):
    """Comment on a Forgejo issue."""

    name = "forgejo_issue_comment"
    description = (
        "Post a comment on a Forgejo issue and return the created comment (JSON). "
        f"'body' is the comment text (Markdown). {_WHERE_ISSUE}"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_ISSUE_PROPERTIES,
            "body": {"type": "string", "description": "Comment text (Markdown)"},
        },
        "required": ["body"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        comment = await comment_on_issue(
            body=arguments.get("body") or "", **self._args(arguments, _ISSUE_KEYS)
        )
        return json.dumps(comment, indent=2)


async def edit_issue_comment(
    *,
    comment_id: int | str,
    body: str,
    instance_url: str | None = None,
    repo: str | None = None,
    issue_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Replace the text of comment ``comment_id`` (``PATCH .../issues/comments/{id}``) and
    return the updated comment. The comment must belong to the named issue."""
    cid = positive_id(comment_id, "comment_id")
    if not isinstance(body, str) or not body.strip():
        raise ForgejoError("comment body must be a non-empty string")
    token = require_token(token, "editing a comment")
    base, slug, number = _resolve_issue(instance_url, repo, issue_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    await ensure_comment_belongs(
        **common, number=number, comment_id=cid, label=f"issue #{number}"
    )
    response = await send(
        **common, path=f"/issues/comments/{cid}", method="PATCH", json_body={"body": body}
    )
    return _json(response)


async def delete_issue_comment(
    *,
    comment_id: int | str,
    instance_url: str | None = None,
    repo: str | None = None,
    issue_number: int | str | None = None,
    url: str | None = None,
    token: SecretValue | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Delete comment ``comment_id`` (``DELETE .../issues/comments/{id}``); it must belong
    to the named issue. Returns ``{"deleted": True, ...}``."""
    cid = positive_id(comment_id, "comment_id")
    token = require_token(token, "deleting a comment")
    base, slug, number = _resolve_issue(instance_url, repo, issue_number, url)
    common: dict[str, Any] = {
        "base": base, "slug": slug, "token": token, "client": client, "timeout": timeout,
    }  # fmt: skip
    await ensure_comment_belongs(
        **common, number=number, comment_id=cid, label=f"issue #{number}"
    )
    await send(**common, path=f"/issues/comments/{cid}", method="DELETE", ok_status=204)
    return {"deleted": True, "comment_id": cid, "issue_number": number}


_COMMENT_ID = {"type": "integer", "description": "Numeric comment id"}


class ForgejoIssueCommentEditTool(ForgejoTool):
    """Edit a comment on a Forgejo issue."""

    name = "forgejo_issue_comment_edit"
    description = (
        "Replace the text of a comment on a Forgejo issue and return the updated comment "
        "(JSON). 'comment_id' is the comment's numeric id (the 'id' returned by "
        "forgejo_issue_comment, or the number in '#issuecomment-N' of its link); it must "
        f"belong to the named issue. 'body' is the new text (Markdown). {_WHERE_ISSUE} "
        "The configured token must be allowed to edit that comment."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            **_ISSUE_PROPERTIES,
            "comment_id": _COMMENT_ID,
            "body": {"type": "string", "description": "New comment text (Markdown)"},
        },
        "required": ["comment_id", "body"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        comment = await edit_issue_comment(
            comment_id=arguments.get("comment_id"),
            body=arguments.get("body") or "",
            **self._args(arguments, _ISSUE_KEYS),
        )
        return json.dumps(comment, indent=2)


class ForgejoIssueCommentDeleteTool(ForgejoTool):
    """Delete a comment from a Forgejo issue."""

    name = "forgejo_issue_comment_delete"
    description = (
        "Delete a comment from a Forgejo issue via the Forgejo API (irreversible). "
        "'comment_id' is the comment's numeric id (the 'id' returned by "
        "forgejo_issue_comment, or the number in '#issuecomment-N' of its link); it must "
        f"belong to the named issue. {_WHERE_ISSUE} The configured token "
        "must be allowed to delete that comment."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {**_ISSUE_PROPERTIES, "comment_id": _COMMENT_ID},
        "required": ["comment_id"],
    }

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await delete_issue_comment(
            comment_id=arguments.get("comment_id"), **self._args(arguments, _ISSUE_KEYS)
        )
        return json.dumps(result)
