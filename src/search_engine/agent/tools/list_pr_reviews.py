"""`list_pr_reviews` tool — list formal reviews on a GitHub PR.

A "review" here is a submission via GitHub's review UI (Approve /
Request changes / Comment), distinct from individual inline review
comments (those live under `list_pr_comments`). Each review has a
state — APPROVED, CHANGES_REQUESTED, COMMENTED, PENDING, DISMISSED —
plus the reviewer's summary body.

Useful when the agent needs to answer "has this PR been approved",
"who blocked it", or "what feedback did the reviewer leave".
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


def _slim(review: dict[str, Any]) -> dict[str, Any]:
    body = review.get("body") or ""
    return {
        "id": review.get("id"),
        "author": (review.get("user") or {}).get("login"),
        "state": review.get("state"),
        "submitted_at": review.get("submitted_at"),
        "html_url": review.get("html_url"),
        "commit_id": review.get("commit_id"),
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

    resp = _github_get(
        account,
        f"/repos/{owner}/{repo}/pulls/{number}/reviews",
        params={"per_page": limit},
    )
    if resp.status_code == 404:
        raise ToolError(f"PR {owner}/{repo}#{number} not found or not accessible.")
    if not resp.ok:
        raise ToolError(f"GitHub API error: {resp.status_code}.")

    raw_reviews = resp.json() or []
    # Newest first — GitHub returns oldest-first by default for reviews.
    raw_reviews.sort(key=lambda r: r.get("submitted_at") or "", reverse=True)
    reviews = [_slim(r) for r in raw_reviews[:limit]]

    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "reviews": reviews,
        "returned_count": len(reviews),
        "page_total": len(raw_reviews),
        "__summary__": (f"Listed {len(reviews)} review(s) on PR {owner}/{repo}#{number}"),
    }


LIST_PR_REVIEWS = Tool(
    name="list_pr_reviews",
    description=(
        "List formal reviews on a GitHub pull request (Approve / "
        "Request changes / Comment submissions via the review UI). "
        "Returns newest first. Each entry includes the reviewer, "
        "state (APPROVED, CHANGES_REQUESTED, COMMENTED, PENDING, "
        "DISMISSED), submitted_at, and the review's summary body. "
        "For individual inline review comments anchored to code lines, "
        "use `list_pr_comments` instead."
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
                    f"Max reviews to return (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
                ),
            },
        },
        "required": ["pr_url"],
    },
    run=_run,
)
