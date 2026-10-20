"""Forgejo tools: read and act on pull requests and issues, and clone repositories.

They are regular tools, available alongside the others unless an agent's ``tools:``
list leaves them out. All talk to a Forgejo instance named per call; the API token
for that instance's host comes from the secret registry (the model never names it).
``forgejo_clone``
writes to disk, so -- like the file tools -- it exists only for an agent with a
workspace.
"""

from __future__ import annotations

from ...secrets import ToolSecrets
from ..base import Tool
from ..workspace import Workspace
from .clone import ForgejoCloneTool
from .issue import (
    ForgejoIssueCommentDeleteTool,
    ForgejoIssueCommentEditTool,
    ForgejoIssueCommentTool,
    ForgejoIssueCreateTool,
    ForgejoIssueListTool,
    ForgejoIssueTool,
    ForgejoIssueUpdateTool,
)
from .pr import (
    ForgejoPRCloseTool,
    ForgejoPRCommentDeleteTool,
    ForgejoPRCommentEditTool,
    ForgejoPRCommentTool,
    ForgejoPRCreateTool,
    ForgejoPRDiffTool,
    ForgejoPRMergeTool,
    ForgejoPRReviewDeleteTool,
    ForgejoPRReviewTool,
    ForgejoPRTool,
)

FORGEJO_TOOL_NAMES: tuple[str, ...] = (
    ForgejoPRTool.name,
    ForgejoPRDiffTool.name,
    ForgejoPRCommentTool.name,
    ForgejoPRCommentEditTool.name,
    ForgejoPRReviewTool.name,
    ForgejoPRCommentDeleteTool.name,
    ForgejoPRReviewDeleteTool.name,
    ForgejoPRCreateTool.name,
    ForgejoPRCloseTool.name,
    ForgejoPRMergeTool.name,
    ForgejoIssueCreateTool.name,
    ForgejoIssueListTool.name,
    ForgejoIssueTool.name,
    ForgejoIssueUpdateTool.name,
    ForgejoIssueCommentTool.name,
    ForgejoIssueCommentEditTool.name,
    ForgejoIssueCommentDeleteTool.name,
    ForgejoCloneTool.name,
)
# The Forgejo tools that need a workspace to confine them to.
FORGEJO_WORKSPACE_TOOL_NAMES: frozenset[str] = frozenset({ForgejoCloneTool.name})


def forgejo_tools(
    workspace: Workspace | None = None, secrets: ToolSecrets | None = None
) -> list[Tool]:
    """The Forgejo tools, in registration order (``forgejo_clone`` only with a
    ``workspace``). ``secrets`` supplies their API tokens; without it they call
    the API anonymously."""
    tools: list[Tool] = [
        ForgejoPRTool(secrets),
        ForgejoPRDiffTool(secrets),
        ForgejoPRCommentTool(secrets),
        ForgejoPRCommentEditTool(secrets),
        ForgejoPRReviewTool(secrets),
        ForgejoPRCommentDeleteTool(secrets),
        ForgejoPRReviewDeleteTool(secrets),
        ForgejoPRCreateTool(secrets),
        ForgejoPRCloseTool(secrets),
        ForgejoPRMergeTool(secrets),
        ForgejoIssueCreateTool(secrets),
        ForgejoIssueListTool(secrets),
        ForgejoIssueTool(secrets),
        ForgejoIssueUpdateTool(secrets),
        ForgejoIssueCommentTool(secrets),
        ForgejoIssueCommentEditTool(secrets),
        ForgejoIssueCommentDeleteTool(secrets),
    ]
    if workspace is not None:
        tools.append(ForgejoCloneTool(workspace, secrets))
    return tools
