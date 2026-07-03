"""Todo item chunker.

Each `ToDoItem` becomes one indexable entity (one chunk per item).
Todos are personal — only the owning user can see them. The chunk
encodes the group's local_date inside `entity_id` so the frontend
can deep-link to the right day without a new mapped field on the
index schema.

Shape:
  entity_type = "todo"
  entity_id   = "todo:<YYYY-MM-DD>:item:<item_id>"
  chunk_type  = "todo_item"
  acl_user_ids = [owner_user_id]
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from django.db.models import Q

from origin.models.chat.todo_models import ToDoItem
from origin.search_engine.chunkers.base import Chunk, EntityChunks, iso, make_snippet
from origin.search_engine.text_extraction import extract_text


def iter_todo_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = ToDoItem.objects.select_related("group", "group__team", "group__user", "category")
    if since is not None:
        # Pick up items whose own row updated, OR whose group updated
        # (e.g. group-level metadata churn).
        qs = qs.filter(Q(ts_updated_at__gte=since) | Q(group__ts_updated_at__gte=since))

    for item in qs:
        group = item.group
        if group is None or not group.team_id or not group.user_id:
            continue

        team_id = str(group.team_id)
        owner_id = str(group.user_id)
        local_date = group.local_date.isoformat()
        entity_id = f"todo:{local_date}:item:{item.item_id}"

        notes_text = extract_text(item.notes) if item.notes else ""
        category_name = item.category.name if item.category_id else ""

        # Build the searchable text. Title + category + notes joined by
        # newlines so OpenSearch's English analyzer treats them as
        # distinct phrases without losing keyword adjacency.
        parts: list[str] = []
        if item.title:
            parts.append(item.title)
        if category_name:
            parts.append(category_name)
        if notes_text:
            parts.append(notes_text)
        search_text = "\n".join(parts).strip()
        if not search_text:
            # Nothing indexable. Skip — saves an embed call.
            continue

        snippet_text = make_snippet(notes_text or item.title)

        # Group the item with its day-level entity via related ids so a
        # future "todo_day_summary" chunker can cross-reference.
        related = [f"todo:{local_date}"]

        chunk = Chunk(
            chunk_id=entity_id,
            entity_type="todo",
            entity_id=entity_id,
            chunk_type="todo_item",
            team_id=team_id,
            acl_user_ids=[owner_id],
            title=item.title or f"Todo {item.item_id}",
            search_text=search_text,
            snippet_text=snippet_text,
            related_entity_ids=related,
            created_at=iso(item.ts_created_at),
            updated_at=iso(item.ts_updated_at),
        )

        yield EntityChunks(
            entity_type="todo",
            entity_id=entity_id,
            chunks=[chunk],
        )
