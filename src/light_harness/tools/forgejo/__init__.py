"""Forgejo tools: read and act on pull requests and issues, and clone repositories.

They are regular tools, available alongside the others unless an agent's ``tools:``
list leaves them out. All talk to a Forgejo instance named per call. ``forgejo_clone``
writes to disk, so -- like the file tools -- it exists only for an agent with a
workspace.
"""

from __future__ import annotations

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


def forgejo_tools(workspace: Workspace | None = None) -> list[Tool]:
    """The Forgejo tools, in registration order (``forgejo_clone`` only with a
    ``workspace``)."""
    tools: list[Tool] = [
        ForgejoPRTool(),
        ForgejoPRDiffTool(),
        ForgejoPRCommentTool(),
        ForgejoPRCommentEditTool(),
        ForgejoPRReviewTool(),
        ForgejoPRCommentDeleteTool(),
        ForgejoPRReviewDeleteTool(),
        ForgejoPRCreateTool(),
        ForgejoPRCloseTool(),
        ForgejoPRMergeTool(),
        ForgejoIssueCreateTool(),
        ForgejoIssueListTool(),
        ForgejoIssueTool(),
        ForgejoIssueUpdateTool(),
        ForgejoIssueCommentTool(),
        ForgejoIssueCommentEditTool(),
        ForgejoIssueCommentDeleteTool(),
    ]
    if workspace is not None:
        tools.append(ForgejoCloneTool(workspace))
    return tools
