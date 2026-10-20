"""forgejo_pr_create / forgejo_pr_close / forgejo_pr_merge: what is sent to the server."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from light_harness.secrets import SecretValue
from light_harness.tools.forgejo.pr import ForgejoError, close_pr, create_pr, merge_pr

REPO = {"instance_url": "https://forge.test", "repo": "o/r", "token": SecretValue("forgejo", "tok")}
API = "/api/v1/repos/o/r"
PR = {"number": 9, "state": "closed", "merged": True, "merge_commit_sha": "abc", "html_url": "u"}


def _client(seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        route = (request.method, request.url.path.removeprefix(API))
        table = {
            ("GET", ""): httpx.Response(200, json={"default_branch": "trunk"}),
            ("POST", "/pulls"): httpx.Response(201, json={"number": 9}),
            ("PATCH", "/pulls/9"): httpx.Response(201, json={"number": 9, "state": "closed"}),
            ("POST", "/pulls/9/merge"): httpx.Response(200),
            ("GET", "/pulls/9"): httpx.Response(200, json=PR),
        }
        return table.get(route, httpx.Response(404, json={"message": f"no route {route}"}))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_create_targets_the_default_branch_unless_a_base_is_given() -> None:
    seen: list[httpx.Request] = []
    asyncio.run(create_pr(title=" Fix ", head="feat", body="b", client=_client(seen), **REPO))

    assert [(r.method, r.url.path.removeprefix(API)) for r in seen] == [
        ("GET", ""),
        ("POST", "/pulls"),
    ]
    assert json.loads(seen[1].content) == {
        "title": "Fix", "head": "feat", "base": "trunk", "body": "b",
    }  # fmt: skip

    seen.clear()
    asyncio.run(create_pr(title="t", head="feat", base="dev", client=_client(seen), **REPO))
    assert [r.method for r in seen] == ["POST"]  # no default-branch lookup
    assert json.loads(seen[0].content)["base"] == "dev"


def test_create_refuses_bad_input_before_any_request() -> None:
    seen: list[httpx.Request] = []
    for kwargs, message in (
        ({"title": " ", "head": "x"}, "title"),
        ({"title": "t", "head": ""}, "head"),
        ({"title": "t", "head": "x", "token": None}, "API token"),
    ):
        with pytest.raises(ForgejoError, match=message):
            asyncio.run(create_pr(client=_client(seen), **{**REPO, **kwargs}))
    assert seen == []


def test_close_sets_state_closed_only() -> None:
    seen: list[httpx.Request] = []
    pr = asyncio.run(close_pr(pr_number=9, client=_client(seen), **REPO))

    assert pr["state"] == "closed"
    assert [(r.method, r.url.path.removeprefix(API)) for r in seen] == [("PATCH", "/pulls/9")]
    assert json.loads(seen[0].content) == {"state": "closed"}


def test_merge_maps_options_and_reports_the_resulting_state() -> None:
    seen: list[httpx.Request] = []
    result = asyncio.run(
        merge_pr(
            pr_number=9, method="squash", title="T", message="M", delete_branch=True,
            head_commit_id="deadbeef", client=_client(seen), **REPO,
        )  # fmt: skip
    )

    assert json.loads(seen[0].content) == {
        "Do": "squash", "MergeTitleField": "T", "MergeMessageField": "M",
        "delete_branch_after_merge": True, "head_commit_id": "deadbeef",
    }  # fmt: skip
    assert result == {
        "merged": True, "state": "closed", "merge_commit_sha": "abc",
        "pr_number": 9, "html_url": "u",
    }  # fmt: skip

    seen.clear()
    with pytest.raises(ForgejoError, match="method must be one of"):
        asyncio.run(merge_pr(pr_number=9, method="yolo", client=_client(seen), **REPO))
    assert seen == []
