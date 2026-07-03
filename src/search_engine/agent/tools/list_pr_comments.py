"""`list_pr_comments` tool — list comments on a GitHub PR.

GitHub splits PR comments across two endpoints:
  * `/repos/{o}/{r}/issues/{n}/comments` — top-level conversation
  * `/repos/{o}/{r}/pulls/{n}/comments`  — inline review comments
    (anchored to a file/line)

We fetch both, merge them, sort newest-first, and cap the response so
the agent doesn't drown in a long-running PR's discussion. Bodies are
wrapped with `wrap_workspace_content` since they're user-authored text.
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
_BODY_CAP = 600


def _slim(comment: dict[str, Any], kind: str) -> dict[str, Any]:
    body = comment.get("body") or ""
    return {
        "id": comment.get("id"),
        "kind": kind,  # "issue" (top-level) or "review" (inline)
        "author": (comment.get("user") or {}).get("login"),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "html_url": comment.get("html_url"),
        # Inline review comments carry file / line context; top-level
        # issue comments don't. Both keys present (null for top-level)
        # keep the shape uniform for the LLM.
        "file_path": comment.get("path"),
        "line": comment.get("line"),
        "body": wrap_workspace_content(body[:_BODY_CAP]),
        "body_truncated": len(body) > _BODY_CAP,
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

    # Both calls are cheap (single page). We over-fetch a little so the
    # merge + cap step has something to pick from when both kinds are
    # active in the same PR.
    fetch_per_kind = min(limit, _MAX_LIMIT)

    issue_resp = _github_get(
        account,
        f"/repos/{owner}/{repo}/issues/{number}/comments",
        params={"per_page": fetch_per_kind},
    )
    if issue_resp.status_code == 404:
        raise ToolError(f"PR {owner}/{repo}#{number} not found or not accessible.")
    if not issue_resp.ok:
        raise ToolError(f"GitHub API error (issue comments): {issue_resp.status_code}.")

    review_resp = _github_get(
        account,
        f"/repos/{owner}/{repo}/pulls/{number}/comments",
        params={"per_page": fetch_per_kind},
    )
    if not review_resp.ok:
        raise ToolError(f"GitHub API error (review comments): {review_resp.status_code}.")

    merged = [_slim(c, "issue") for c in (issue_resp.json() or [])] + [
        _slim(c, "review") for c in (review_resp.json() or [])
    ]
    merged.sort(key=lambda c: c.get("created_at") or "", reverse=True)
    capped = merged[:limit]

    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "comments": capped,
        "returned_count": len(capped),
        "total_known": len(merged),
        "__summary__": (
            f"Listed {len(capped)} comment(s) on PR {owner}/{repo}#{number}"
            + (f" (of {len(merged)} known)" if len(merged) > len(capped) else "")
        ),
    }


LIST_PR_COMMENTS = Tool(
    name="list_pr_comments",
    description=(
        "List comments on a GitHub pull request — both top-level "
        "conversation comments AND inline review comments anchored to "
        "code lines. Returns newest first, capped at `limit` (default "
        f"{_DEFAULT_LIMIT}, max {_MAX_LIMIT}). Each entry includes "
        "author, body, created_at, html_url, plus file_path + line for "
        "inline comments. Use this to read the discussion on a PR — for "
        "the PR's structural metadata (title, state, diff stats), call "
        "`fetch_pr` instead."
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
                    f"Max comments to return (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
                ),
            },
        },
        "required": ["pr_url"],
    },
    run=_run,
)
