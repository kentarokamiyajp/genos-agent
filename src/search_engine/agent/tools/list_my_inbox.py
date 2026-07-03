"""`list_my_inbox` tool — recent items from the caller's inbox.

Surfaces the `InboxItems` model that backs the in-app `/workspace/inbox`
route. Two main item shapes:

  * Activities (`item_type=0`) — passive notifications, e.g. "Alice
    closed task WRD-5", "Bob commented on your task".
  * Requests (`item_type ∈ {1, 2, 3}`) — actionable join requests:
    1=join team, 2=join project, 3=join GM. Each carries
    `request_status` ("pending" / "approved" / "rejected").

Use `category` to scope. Default is `all` so a single call answers
"what's in my inbox?".

ACL contract:
  * Tenant guard: ctx.team_id.
  * Scope: `receiver_id == ctx.user_id` (server-trusted). No way for
    the LLM to surface another user's inbox.
  * Soft-delete tombstones (`is_deleted=True`) excluded.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.inbox_models import InboxItems
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_LIMIT = 50
_DEFAULT_LIMIT = 20
_VALID_CATEGORIES = {"activity", "requests", "all"}

# Mirror of the mapping comment inside InboxItems.item_type (origin/models/
# common/inbox_models.py:30-37). Kept here so the tool response is
# self-descriptive — agents don't have to know the numeric code.
_ITEM_TYPE_LABELS = {
    0: "activity",
    1: "join_team_request",
    2: "join_project_request",
    3: "join_gm_request",
}
_REQUEST_ITEM_TYPES = (1, 2, 3)


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    category = args.get("category", "all")
    if category not in _VALID_CATEGORIES:
        raise ToolError(
            f"`category` must be one of {sorted(_VALID_CATEGORIES)} (got {category!r})."
        )

    include_read = bool(args.get("include_read", False))

    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        raise ToolError(f"`limit` must be an integer (got {args.get('limit')!r}).")
    limit = max(1, min(limit, _MAX_LIMIT))

    qs = InboxItems.objects.select_related("sender").filter(
        team_id=ctx.team_id,
        receiver_id=ctx.user_id,
        is_deleted=False,
    )
    if category == "activity":
        qs = qs.filter(item_type=0)
    elif category == "requests":
        qs = qs.filter(item_type__in=_REQUEST_ITEM_TYPES)
    if not include_read:
        qs = qs.filter(is_read=False)

    qs = qs.order_by("-ts_created_at")[:limit]

    items: list[dict[str, Any]] = []
    for row in qs:
        sender = row.sender
        items.append(
            {
                "item_id": row.item_id,
                "item_type": row.item_type,
                "item_type_label": _ITEM_TYPE_LABELS.get(row.item_type, f"type_{row.item_type}"),
                "sender_id": str(sender.id) if sender else None,
                "sender_username": sender.username if sender else None,
                "is_read": row.is_read,
                "request_status": (
                    row.request_status if row.item_type in _REQUEST_ITEM_TYPES else None
                ),
                "item_body": row.item_body,
                "ts_created_at": row.ts_created_at.isoformat(),
            }
        )

    if items:
        unread_n = sum(1 for it in items if not it["is_read"])
        pending_requests = sum(
            1
            for it in items
            if it["item_type"] in _REQUEST_ITEM_TYPES and it.get("request_status") == "pending"
        )
        summary_bits = [f"{len(items)} inbox item(s)"]
        if not include_read:
            summary_bits[0] = f"{len(items)} unread item(s)"
        elif unread_n:
            summary_bits.append(f"{unread_n} unread")
        if pending_requests:
            summary_bits.append(f"{pending_requests} pending request(s)")
        summary = "; ".join(summary_bits)
    else:
        summary = "No inbox items match." if include_read else "No unread inbox items."

    return {
        "items": items,
        "category": category,
        "include_read": include_read,
        "__summary__": summary,
    }


LIST_MY_INBOX = Tool(
    name="list_my_inbox",
    description=(
        "Recent items from the current user's inbox (`InboxItems`). Two "
        "broad kinds: activity notifications (item_type=0, e.g. 'Alice "
        "closed task WRD-5') and join requests (item_type 1/2/3 — team/"
        "project/GM, each with request_status 'pending'/'approved'/"
        "'rejected'). Use for 'what's in my inbox?', 'any notifications "
        "for me?', 'do I have pending requests?'. Returns items sorted "
        "newest-first. By default only unread items are returned; pass "
        "`include_read=true` to include items already marked read. Pass "
        "`category` to scope to 'activity' or 'requests'. Caller's inbox "
        "only — there is no way to specify another user."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "category": {
                "type": "STRING",
                "enum": ["activity", "requests", "all"],
                "description": (
                    "Scope to activity notifications, join requests, or " "both. Default 'all'."
                ),
            },
            "include_read": {
                "type": "BOOLEAN",
                "description": (
                    "If true, include items already marked read. Default " "false (only unread)."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max items to return (1–{_MAX_LIMIT}). Default " f"{_DEFAULT_LIMIT}."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
