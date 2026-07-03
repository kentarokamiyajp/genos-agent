"""`list_today_todos` — read-only tool.

Returns the requesting user's todos for today (server-local date).
ACL is implicit: filters by `group.user_id == ctx.user_id`.

Note(tz): the server-local date is used as a proxy for the user's
"today". Multi-region usage would require a per-user timezone
preference; flagged for the deferred Phase 4 work.
"""

from __future__ import annotations

from typing import Any

from django.utils import timezone

from origin.models.chat.todo_models import ToDoGroup
from origin.search_engine.agent.tools.base import Tool, ToolContext


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    today = timezone.localdate()
    group = (
        ToDoGroup.objects.filter(user_id=ctx.user_id, local_date=today)
        .prefetch_related("items", "items__category")
        .first()
    )
    if group is None:
        return {
            "local_date": today.isoformat(),
            "group_id": None,
            "is_completed": False,
            "items": [],
            "__summary__": "No todos for today yet.",
        }

    items = []
    completed = 0
    for item in group.items.all().order_by("sort_order", "item_id"):
        if item.is_completed:
            completed += 1
        items.append(
            {
                "item_id": item.item_id,
                "title": item.title,
                "is_completed": item.is_completed,
                "category_name": item.category.name if item.category_id else None,
            }
        )

    total = len(items)
    return {
        "local_date": today.isoformat(),
        "group_id": group.group_id,
        "is_completed": group.is_completed,
        "items": items,
        "__summary__": f"Today: {completed} of {total} done.",
    }


LIST_TODAY_TODOS = Tool(
    name="list_today_todos",
    description=(
        "List the user's todo items for today (the current calendar day). "
        "Returns each item with title, completion status, and optional "
        "category. Use this when the user asks 'what's on my list today?', "
        "'what's left for today?', or similar. Read-only — does not require "
        "approval."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
    },
    run=_run,
    requires_approval=False,
)
