"""`list_pr_files` tool — list files changed in a GitHub PR.

Returns per-file: filename, status (added/modified/removed/renamed),
additions, deletions. Patch content is intentionally NOT included —
even a moderately-sized PR's combined patch blows the agent's context
budget, and the agent rarely needs to read raw diff bytes to answer
"what did this PR change". If the agent does need code, it should
follow up with a targeted code-search tool or a future `fetch_pr_patch`
tool that scopes to a single file.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.user_models import CustomUser
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.services.github_webhooks import parse_pr_url_full
from origin.views.common.github_views import _connected_account, _github_get

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 100


def _slim(file: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": file.get("filename"),
        "previous_filename": file.get("previous_filename"),  # set on renames
        "status": file.get("status"),
        "additions": file.get("additions"),
        "deletions": file.get("deletions"),
        "changes": file.get("changes"),
        "blob_url": file.get("blob_url"),
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
        f"/repos/{owner}/{repo}/pulls/{number}/files",
        params={"per_page": limit},
    )
    if resp.status_code == 404:
        raise ToolError(f"PR {owner}/{repo}#{number} not found or not accessible.")
    if not resp.ok:
        raise ToolError(f"GitHub API error: {resp.status_code}.")

    raw_files = resp.json() or []
    files = [_slim(f) for f in raw_files][:limit]
    total_additions = sum((f.get("additions") or 0) for f in raw_files)
    total_deletions = sum((f.get("deletions") or 0) for f in raw_files)

    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "files": files,
        "returned_count": len(files),
        "page_total": len(raw_files),
        "totals": {
            "additions": total_additions,
            "deletions": total_deletions,
        },
        "__summary__": (
            f"Listed {len(files)} file(s) changed in PR {owner}/{repo}#{number} "
            f"(+{total_additions}/−{total_deletions})"
        ),
    }


LIST_PR_FILES = Tool(
    name="list_pr_files",
    description=(
        "List files changed in a GitHub pull request with per-file "
        "status (added/modified/removed/renamed) and additions/deletions "
        "counts. Patch content is NOT included to keep the response "
        "compact. Use this to see the shape of a PR's changes — for "
        "the conversation around it, call `list_pr_comments`."
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
                    f"Max files to return (default {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
                ),
            },
        },
        "required": ["pr_url"],
    },
    run=_run,
)
