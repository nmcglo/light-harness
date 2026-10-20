"""Forgejo issue tools: what is sent to the server and what comes back."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from light_harness.secrets import SecretValue
from light_harness.tools.forgejo.common import ForgejoError
from light_harness.tools.forgejo.issue import (
    create_issue,
    get_issue,
    list_issues,
    parse_issue_url,
    update_issue,
)

REPO = {"instance_url": "https://forge.test", "repo": "o/r", "token": SecretValue("forgejo", "tok")}
API = "https://forge.test/api/v1/repos/o/r"
LABELS = [{"id": 11, "name": "bug"}, {"id": 12, "name": "docs"}]
ISSUE = {"number": 3, "title": "t", "state": "open", "labels": [], "assignees": []}


def _client(
    seen: list[httpx.Request], routes: dict[tuple[str, str], httpx.Response] | None = None
) -> httpx.AsyncClient:
    table = {
        ("GET", "/labels"): httpx.Response(200, json=LABELS),
        ("POST", "/issues"): httpx.Response(201, json=ISSUE),
        ("PATCH", "/issues/3"): httpx.Response(201, json=ISSUE),
        ("PUT", "/issues/3/labels"): httpx.Response(200, json=LABELS[:1]),
        ("GET", "/issues/3"): httpx.Response(200, json=ISSUE),
        ("GET", "/issues/3/comments"): httpx.Response(200, json=[{"id": 1, "body": "hi"}]),
        **(routes or {}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = (request.method, str(request.url).removeprefix(API).split("?")[0])
        return table.get(key, httpx.Response(404, json={"message": f"no route {key}"}))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro_fn: Callable[..., object], seen: list[httpx.Request], **kwargs: object):
    return asyncio.run(coro_fn(client=_client(seen), **REPO, **kwargs))


def test_label_names_are_sent_as_ids_and_unknown_ones_stop_before_creating() -> None:
    seen: list[httpx.Request] = []
    _run(create_issue, seen, title=" Crash ", body="b", labels=["docs", "bug"], assignees=["al"])
    post = next(r for r in seen if r.method == "POST")
    assert json.loads(post.content) == {
        "title": "Crash", "body": "b", "labels": [12, 11], "assignees": ["al"],
    }  # fmt: skip

    seen.clear()
    with pytest.raises(ForgejoError, match=r"unknown label\(s\): nope; the repo has: bug, docs"):
        _run(create_issue, seen, title="x", labels=["nope"])
    assert [r.method for r in seen] == ["GET"]  # nothing was created


def test_update_changes_only_what_was_passed_in_order_and_returns_the_fresh_issue() -> None:
    seen: list[httpx.Request] = []
    _run(update_issue, seen, issue_number=3, state="closed", labels=["bug"])

    assert [(r.method, r.url.path.removeprefix("/api/v1/repos/o/r")) for r in seen] == [
        ("GET", "/labels"),
        ("PATCH", "/issues/3"),
        ("PUT", "/issues/3/labels"),
        ("GET", "/issues/3"),
    ]
    assert json.loads(seen[1].content) == {"state": "closed"}
    assert json.loads(seen[2].content) == {"labels": [11]}

    with pytest.raises(ForgejoError, match="nothing to update"):
        _run(update_issue, [], issue_number=3)
    with pytest.raises(ForgejoError, match="state must be open or closed"):
        _run(update_issue, [], issue_number=3, state="done")


def test_list_asks_for_issues_only_and_summarises() -> None:
    seen: list[httpx.Request] = []
    full = {**ISSUE, "user": {"login": "al"}, "labels": [{"name": "bug"}], "body": "long text"}
    routes = {("GET", "/issues"): httpx.Response(200, json=[full], headers={"X-Total-Count": "7"})}
    result = asyncio.run(
        list_issues(
            client=_client(seen, routes), state="all", labels=["bug", "docs"], query="crash",
            limit=5, **REPO,
        )  # fmt: skip
    )

    params = dict(seen[0].url.params)
    assert params == {
        "state": "all", "type": "issues", "limit": "5", "page": "1",
        "labels": "bug,docs", "q": "crash",
    }  # fmt: skip
    assert result["total"] == 7
    assert result["issues"][0]["user"] == "al" and result["issues"][0]["labels"] == ["bug"]
    assert "body" not in result["issues"][0]


def test_issue_url_and_comments() -> None:
    assert parse_issue_url("https://h/sub/o/r/issues/3") == ("https://h/sub", "o/r", 3)
    assert parse_issue_url("https://h/o/r/pulls/9/") == ("https://h", "o/r", 9)
    with pytest.raises(ForgejoError, match="not a Forgejo issue URL"):
        parse_issue_url("https://h/o/r")

    seen: list[httpx.Request] = []
    issue = asyncio.run(
        get_issue(
            client=_client(seen), url="https://forge.test/o/r/issues/3", include_comments=True
        )
    )
    assert issue["comment_list"] == [{"id": 1, "body": "hi"}]
