"""`list_pr_commits` tool — list commits on a GitHub PR.

Returns slim records: short sha, author (login or commit author name),
first line of the commit message, and the commit date. Full message
bodies / patches are intentionally omitted — they're rarely useful at
this resolution and inflate the response.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.user_models import CustomUser
from origin.search_engine.agent.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    wrap_workspace_content,
)
from origin.services.github_webhooks import parse_pr_url_full
from origin.views.common.github_views import _connected_account, _github_get

_DEFAULT_LIMIT = 30
_MAX_LIMIT = 100


def _slim(commit: dict[str, Any]) -> dict[str, Any]:
    sha = commit.get("sha") or ""
    inner = commit.get("commit") or {}
    message = inner.get("message") or ""
    first_line = message.split("\n", 1)[0]
    # Prefer the resolved GitHub user (login) when available — falls
    # back to the git commit's author name for commits whose email
    # doesn't map to a GitHub account.
    author = (commit.get("author") or {}).get("login") or (inner.get("author") or {}).get("name")
    committed_at = (inner.get("committer") or {}).get("date") or (inner.get("author") or {}).get(
        "date"
    )
    return {
        "sha": sha[:7],
        "full_sha": sha,
        "author": author,
        "message_first_line": wrap_workspace_content(first_line),
        "committed_at": committed_at,
        "html_url": commit.get("html_url"),
    }


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    pr_url = args.get("pr_url")
    if not isinstance(pr_url, str) or not pr_url:
        raise ToolError("pr_url is required (full GitHub PR URL).")
    ref = parse_pr_url_full(pr_url)
    if ref is None:
        raise ToolError(f"Invalid PR URL: {pr_url!r}")
    owner, repo, number = ref

    raw_limit = args.get("limit", _DEFAULT_LIMIT)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        raise ToolError(f"limit must be an integer (got {raw_limit!r}).")
    limit = max(1, min(limit, _MAX_LIMIT))

    try:
        user = CustomUser.objects.get(id=ctx.user_id)
    except CustomUser.DoesNotExist:
        raise ToolError("Current user record not found.")

    account = _connected_account(user)
    if account is None:
        raise ToolError("GitHub is not connected for this user.")

    resp = _github_get(
        account,
        f"/repos/{owner}/{repo}/pulls/{number}/commits",
        params={"per_page": limit},
    )
    if resp.status_code == 404:
        raise ToolError(f"PR {owner}/{repo}#{number} not found or not accessible.")
    if not resp.ok:
        raise ToolError(f"GitHub API error: {resp.status_code}.")

    raw_commits = resp.json() or []
    commits = [_slim(c) for c in raw_commits][:limit]

    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "commits": commits,
        "returned_count": len(commits),
        "page_total": len(raw_commits),
        "__summary__": (f"Listed {len(commits)} commit(s) on PR {owner}/{repo}#{number}"),
    }


LIST_PR_COMMITS = Tool(
    name="list_pr_commits",
    description=(
        "List commits on a GitHub pull request: short sha, author, "
        "first line of the commit message, and date. Use this to see "
        "the history of a PR — what changes landed in what order. For "
        "structural metadata (state, diff stats), use `fetch_pr` instead."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "pr_url": {
                "type": "STRING",
                "description": "Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/42).",
            },
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max commits to return (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
                ),
            },
        },
        "required": ["pr_url"],
    },
    run=_run,
)
