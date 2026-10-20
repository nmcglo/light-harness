"""Editing and deleting PR/issue comments and PR reviews: what reaches the server."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from light_harness.secrets import SecretValue
from light_harness.tools.forgejo.issue import delete_issue_comment, edit_issue_comment
from light_harness.tools.forgejo.pr import (
    ForgejoError,
    delete_pr_comment,
    delete_pr_review,
    edit_pr_comment,
)

REPO = {"instance_url": "https://forge.test", "repo": "o/r", "token": SecretValue("forgejo", "tok")}
API = "https://forge.test/api/v1/repos/o/r"


def _client(requests: list[httpx.Request], comment_pr: int = 9) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            issue_url = f"https://forge.test/o/r/issues/{comment_pr}"
            return httpx.Response(200, json={"id": 7, "issue_url": issue_url})
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": 7, **json.loads(request.content)})
        return httpx.Response(204)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_comment_is_deleted_when_it_belongs_to_the_pr() -> None:
    seen: list[httpx.Request] = []
    result = asyncio.run(
        delete_pr_comment(comment_id=7, pr_number=9, client=_client(seen), **REPO)
    )

    assert result == {"deleted": True, "comment_id": 7, "pr_number": 9}
    assert [(r.method, str(r.url)) for r in seen] == [
        ("GET", f"{API}/issues/comments/7"),
        ("DELETE", f"{API}/issues/comments/7"),
    ]
    assert all(r.headers["Authorization"] == "token tok" for r in seen)


def test_comment_from_another_pr_is_not_deleted() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(ForgejoError, match="does not belong to PR #9"):
        asyncio.run(
            delete_pr_comment(comment_id=7, pr_number=9, client=_client(seen, comment_pr=3), **REPO)
        )

    assert [r.method for r in seen] == ["GET"]  # nothing was deleted


def test_review_delete_hits_the_review_endpoint_and_needs_a_token() -> None:
    seen: list[httpx.Request] = []
    result = asyncio.run(
        delete_pr_review(review_id="12", pr_number=9, client=_client(seen), **REPO)
    )

    assert result == {"deleted": True, "review_id": 12, "pr_number": 9}
    assert [(r.method, str(r.url)) for r in seen] == [("DELETE", f"{API}/pulls/9/reviews/12")]

    with pytest.raises(ForgejoError, match="requires an API token"):
        asyncio.run(
            delete_pr_review(
                review_id=12, pr_number=9, instance_url="https://forge.test", repo="o/r"
            )
        )


def test_edit_checks_ownership_first_then_patches_only_the_body() -> None:
    seen: list[httpx.Request] = []
    comment = asyncio.run(
        edit_pr_comment(comment_id=7, body="new", pr_number=9, client=_client(seen), **REPO)
    )

    assert comment["body"] == "new"
    assert [(r.method, str(r.url)) for r in seen] == [
        ("GET", f"{API}/issues/comments/7"),
        ("PATCH", f"{API}/issues/comments/7"),
    ]
    assert json.loads(seen[1].content) == {"body": "new"}


def test_edit_of_a_comment_on_another_pr_or_with_an_empty_body_sends_no_patch() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(ForgejoError, match="does not belong to PR #9"):
        asyncio.run(
            edit_pr_comment(
                comment_id=7, body="new", pr_number=9, client=_client(seen, comment_pr=3), **REPO
            )
        )
    with pytest.raises(ForgejoError, match="non-empty"):
        asyncio.run(
            edit_pr_comment(comment_id=7, body="  ", pr_number=9, client=_client(seen), **REPO)
        )
    assert [r.method for r in seen] == ["GET"]


def test_issue_comment_edit_and_delete_are_confined_to_the_named_issue() -> None:
    seen: list[httpx.Request] = []
    client = _client(seen, comment_pr=3)
    edited = asyncio.run(
        edit_issue_comment(comment_id=7, body="new", issue_number=3, client=client, **REPO)
    )
    deleted = asyncio.run(delete_issue_comment(comment_id=7, issue_number=3, client=client, **REPO))
    assert edited["body"] == "new"
    assert deleted == {"deleted": True, "comment_id": 7, "issue_number": 3}

    seen.clear()
    with pytest.raises(ForgejoError, match="does not belong to issue #4"):
        asyncio.run(delete_issue_comment(comment_id=7, issue_number=4, client=client, **REPO))
    assert [r.method for r in seen] == ["GET"]  # the comment on #3 was not deleted
