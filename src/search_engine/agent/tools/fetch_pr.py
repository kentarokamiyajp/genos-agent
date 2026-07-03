"""`fetch_pr` tool — load metadata for one GitHub pull request.

Uses the calling user's OAuth-stored GitHub token (resolved from
`ctx.user_id`) to call the live GitHub API — there's no internal mirror
of PR data, the integration is read-mostly and stateless.

Returns a slim, LLM-friendly summary: title, state (open/draft/merged/
closed), branches, diff stats, dates, author, and a truncated body. PR
body is wrapped with `wrap_workspace_content` since it's user-authored
text that could in theory contain prompt-injection payloads.
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

# Bodies on GitHub PRs are free-form Markdown and occasionally hold
# pages of release-notes / acceptance-criteria text. Cap so we don't
# eat the agent's context budget on a single tool call.
_BODY_CAP = 1000


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    pr_url = args.get("pr_url")
    if not isinstance(pr_url, str) or not pr_url:
        raise ToolError("pr_url is required (full GitHub PR URL).")
    ref = parse_pr_url_full(pr_url)
    if ref is None:
        raise ToolError(f"Invalid PR URL: {pr_url!r}")
    owner, repo, number = ref

    try:
        user = CustomUser.objects.get(id=ctx.user_id)
    except CustomUser.DoesNotExist:
        raise ToolError("Current user record not found.")

    account = _connected_account(user)
    if account is None:
        raise ToolError("GitHub is not connected for this user.")

    resp = _github_get(account, f"/repos/{owner}/{repo}/pulls/{number}")
    if resp.status_code == 404:
        raise ToolError(f"PR {owner}/{repo}#{number} not found or not accessible.")
    if not resp.ok:
        raise ToolError(f"GitHub API error: {resp.status_code}.")
    pr = resp.json() or {}

    body = pr.get("body") or ""
    body_truncated = len(body) > _BODY_CAP
    body_excerpt = body[:_BODY_CAP]

    state = "merged" if pr.get("merged") else "draft" if pr.get("draft") else pr.get("state")

    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "title": pr.get("title") or "",
        "state": state,
        "merged_at": pr.get("merged_at"),
        "html_url": pr.get("html_url"),
        "author": (pr.get("user") or {}).get("login"),
        "head_ref": (pr.get("head") or {}).get("ref"),
        "base_ref": (pr.get("base") or {}).get("ref"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "commits_count": pr.get("commits"),
        "comments_count": (pr.get("comments") or 0) + (pr.get("review_comments") or 0),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "body": wrap_workspace_content(body_excerpt),
        "body_truncated": body_truncated,
        "__summary__": f"Fetched PR {owner}/{repo}#{number}",
    }


FETCH_PR = Tool(
    name="fetch_pr",
    description=(
        "Load metadata for one GitHub pull request: title, state "
        "(open/draft/merged/closed), branches (head → base), diff stats "
        "(additions/deletions/changed_files), commit/comment counts, "
        "dates, author, and a truncated body. Use this when the user "
        "asks about a specific PR by URL — for deeper detail, follow "
        "up with `list_pr_comments`, `list_pr_files`, `list_pr_reviews`, "
        "or `list_pr_commits`. Requires the user to have connected "
        "GitHub via the Integrations page."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "pr_url": {
                "type": "STRING",
                "description": (
                    "Full GitHub PR URL " "(e.g. https://github.com/owner/repo/pull/42)."
                ),
            },
        },
        "required": ["pr_url"],
    },
    run=_run,
)
