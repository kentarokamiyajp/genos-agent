"""`update_todo_item` — write tool, requires user approval.

Patch fields on an existing todo item the user owns. Common use:
toggle `is_completed`, edit `title`, change `category`, set / clear
`notes_text`.

ACL: the requesting user must own the group the item belongs to.
"""

from __future__ import annotations

from typing import Any

from django.db import transaction
from django.utils import timezone

from origin.models.chat.todo_models import ToDoCategory, ToDoGroup, ToDoItem
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _wrap_notes(text: str) -> list[dict[str, Any]] | None:
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


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_id = args.get("item_id")
    try:
        item_id = int(raw_id)
    except (TypeError, ValueError):
        raise ToolError(f"`item_id` must be an integer (got {raw_id!r}).")

    try:
        item = ToDoItem.objects.select_related("group").get(item_id=item_id)
    except ToDoItem.DoesNotExist:
        raise ToolError(f"Todo item {item_id} not found.")

    if str(item.group.user_id) != ctx.user_id:
        raise ToolError("Not authorized to update this todo item.")
    if str(item.group.team_id or "") != ctx.team_id:
        raise ToolError("Todo item belongs to a different team.")

    with transaction.atomic():
        # Title
        if "title" in args and args["title"] is not None:
            new_title = (args["title"] or "").strip()
            if not new_title:
                raise ToolError("`title`, if provided, must not be blank.")
            item.title = new_title

        # Notes — replace with wrapped paragraphs. Pass an empty string
        # to clear notes entirely.
        if "notes_text" in args:
            item.notes = _wrap_notes((args.get("notes_text") or "").strip())

        # Completion
        if "is_completed" in args and args["is_completed"] is not None:
            new_completed = bool(args["is_completed"])
            if new_completed and not item.is_completed:
                item.ts_completed_at = timezone.now()
            elif not new_completed:
                item.ts_completed_at = None
            item.is_completed = new_completed

        # Category (free-form). `clear_category=true` removes it
        # explicitly; passing a non-empty `category` overrides any clear.
        if args.get("clear_category"):
            item.category = None
        category_name = (args.get("category") or "").strip()
        if category_name:
            category, _ = ToDoCategory.objects.get_or_create(
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                name=category_name,
            )
            item.category = category

        item.save()

        # Recompute the group's cached completion.
        has_open = ToDoItem.objects.filter(group=item.group, is_completed=False).exists()
        ToDoGroup.objects.filter(group_id=item.group_id).update(is_completed=not has_open)

    item.refresh_from_db()
    return {
        "item_id": item.item_id,
        "group_id": item.group_id,
        "local_date": item.group.local_date.isoformat(),
        "title": item.title,
        "is_completed": item.is_completed,
        "category": item.category.name if item.category_id else None,
        "__summary__": f"Updated todo {item.item_id}: {item.title}",
    }


UPDATE_TODO_ITEM = Tool(
    name="update_todo_item",
    description=(
        "Update fields on one of the user's existing todo items. REQUIRES "
        "USER APPROVAL. Common uses: mark complete/incomplete, edit title, "
        "change category, replace or clear the notes body. Required: "
        "item_id. Optional patch fields: title, notes_text (empty string "
        "clears it), is_completed, category (free-form), clear_category "
        "(set true to remove the existing category)."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "item_id": {
                "type": "INTEGER",
                "description": "ID of the todo item to update.",
            },
            "title": {
                "type": "STRING",
                "description": "New title. Must be non-blank if provided.",
            },
            "notes_text": {
                "type": "STRING",
                "description": (
                    "Replace the item's notes body with this plain text. "
                    "Pass an empty string to clear notes."
                ),
            },
            "is_completed": {
                "type": "BOOLEAN",
                "description": "Mark the item complete (true) or reopen it (false).",
            },
            "category": {
                "type": "STRING",
                "description": (
                    "Free-form category label. Existing matched "
                    "case-sensitively; otherwise created."
                ),
            },
            "clear_category": {
                "type": "BOOLEAN",
                "description": (
                    "Set to true to remove the existing category. Ignored "
                    "if `category` is also provided."
                ),
            },
        },
        "required": ["item_id"],
    },
    run=_run,
    requires_approval=True,
)
