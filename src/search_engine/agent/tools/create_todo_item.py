"""`create_todo_item` — write tool, requires user approval.

Adds an item to the user's todo group for `target_date` (default today).
Creates the day's group lazily if it doesn't exist yet. `category` is
free-form text — get_or_create on (team, user, name).
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime
from typing import Any

from django.db import transaction
from django.utils import timezone

from origin.models.chat.todo_models import ToDoCategory, ToDoGroup, ToDoItem
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _wrap_notes(text: str) -> list[dict[str, Any]] | None:
    """Wrap plain text into a minimal BlockNote paragraph list, matching
    the shape `extract_text` walks during indexing.
    """
    if not text:
        return None
    return [
        {
            "type": "paragraph",
            "props": {
                "textColor": "default",
                "textAlignment": "left",
                "backgroundColor": "default",
            },
            "content": [{"type": "text", "text": line, "styles": {}}] if line else [],
            "children": [],
        }
        for line in text.split("\n")
    ]


def _parse_date(value, default):
    if not value:
        return default
    if isinstance(value, date_cls) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise ToolError(f"`target_date` must be ISO date YYYY-MM-DD (got {value!r}).")


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    title = (args.get("title") or "").strip()
    if not title:
        raise ToolError("`title` is required.")

    target_date = _parse_date(args.get("target_date"), timezone.localdate())
    notes_text = (args.get("notes_text") or "").strip()
    category_name = (args.get("category") or "").strip()

    with transaction.atomic():
        group, _ = ToDoGroup.objects.get_or_create(
            team_id=ctx.team_id,
            user_id=ctx.user_id,
            local_date=target_date,
            defaults={"is_completed": False},
        )
        category = None
        if category_name:
            category, _ = ToDoCategory.objects.get_or_create(
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                name=category_name,
            )
        item = ToDoItem.objects.create(
            group=group,
            category=category,
            title=title,
            notes=_wrap_notes(notes_text),
        )
        # Newly created incomplete item flips the group's completion
        # back to False if it had been all-done before.
        has_open = ToDoItem.objects.filter(group=group, is_completed=False).exists()
        ToDoGroup.objects.filter(group_id=group.group_id).update(is_completed=not has_open)

    return {
        "item_id": item.item_id,
        "group_id": group.group_id,
        "local_date": target_date.isoformat(),
        "title": item.title,
        "category": category.name if category else None,
        "__summary__": f"Created todo for {target_date.isoformat()}: {item.title}",
    }


CREATE_TODO_ITEM = Tool(
    name="create_todo_item",
    description=(
        "Create a new todo item for the user. REQUIRES USER APPROVAL — the "
        "user will see your proposed arguments and decide whether to "
        "execute. Use this when the user asks to add / create / file a "
        "todo. Required: title. Optional: notes_text (plain-text "
        "description), category (free-form label; created if new), "
        "target_date (ISO YYYY-MM-DD; default today)."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "Short todo title (1 line).",
            },
            "notes_text": {
                "type": "STRING",
                "description": "Optional plain-text body for the todo.",
            },
            "category": {
                "type": "STRING",
                "description": (
                    "Optional free-form category label. Existing label "
                    "matched case-sensitively; otherwise created."
                ),
            },
            "target_date": {
                "type": "STRING",
                "description": ("Optional ISO YYYY-MM-DD date. Default is today (server-local)."),
            },
        },
        "required": ["title"],
    },
    run=_run,
    requires_approval=True,
)
