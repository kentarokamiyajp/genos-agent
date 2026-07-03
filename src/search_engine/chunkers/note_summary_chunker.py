"""Chunker for `NoteSummary` rows.

One chunk per note summary. ACL is derived from the underlying note's
parent context (chat members for chat notes, project members for task
notes, owner-only for personal notes) — same logic the `note_chunker`
uses, so the index view matches the live fetch view.

Entity_type is `"note_summary"` so the wider Spotlight search can
distinguish note summaries from note sections (`entity_type="note"`)
and rank them separately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from origin.models.note.chat_note_models import ChatNoteMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.search_engine.agent.acl import (
    chat_note_acl_user_ids,
    personal_note_acl_user_ids,
    task_note_acl_user_ids,
)
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_LABEL,
    NOTE_TYPE_CHAT,
    NOTE_TYPE_LABEL,
    NOTE_TYPE_PERSONAL,
    NOTE_TYPE_TASK,
    Chunk,
    EntityChunks,
    iso,
    make_snippet,
)
from origin.search_engine.models import NoteSummary


def _resolve_note_for_summary(summary: NoteSummary):
    """Load the parent note row so we can re-derive its ACL.

    Returns `(note_or_None, acl_user_ids_or_empty_set)`. A `None` note
    means the underlying note was deleted after the summary was created
    — in that case the chunker skips emitting (the summary becomes
    unreachable via ACL anyway).
    """
    if summary.note_type == NOTE_TYPE_PERSONAL:
        try:
            note = PersonalNoteMaster.objects.get(note_id=summary.note_id)
        except PersonalNoteMaster.DoesNotExist:
            return None, set()
        owner_id = getattr(note, "owner_id", None)
        acl = personal_note_acl_user_ids(owner_id=owner_id, note_id=summary.note_id)
        return note, acl
    if summary.note_type == NOTE_TYPE_TASK:
        try:
            note = TaskNoteMaster.objects.get(note_id=summary.note_id)
        except TaskNoteMaster.DoesNotExist:
            return None, set()
        owner_id = getattr(note, "owner_id", None)
        project_id = getattr(note, "project_id", None)
        acl = task_note_acl_user_ids(
            owner_id=owner_id, project_id=project_id, note_id=summary.note_id
        )
        return note, acl
    if summary.note_type == NOTE_TYPE_CHAT:
        try:
            note = ChatNoteMaster.objects.get(note_id=summary.note_id)
        except ChatNoteMaster.DoesNotExist:
            return None, set()
        owner_id = getattr(note, "owner_id", None)
        acl = chat_note_acl_user_ids(
            owner_id=owner_id,
            chat_type_code=note.chat_type,
            channel_id=note.channel_id,
            note_id=summary.note_id,
        )
        return note, acl
    return None, set()


def iter_note_summary_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = NoteSummary.objects.all().order_by("id")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)

    for summary in qs.iterator():
        if not summary.team_id or not summary.summary_text:
            continue

        note_label = NOTE_TYPE_LABEL.get(summary.note_type)
        if not note_label:
            continue

        note, acl_set = _resolve_note_for_summary(summary)
        # Underlying note gone, or ACL empty → not reachable by anyone.
        # Defensive skip (matches the thread-summary chunker pattern).
        if note is None or not acl_set:
            continue

        acl_user_ids = sorted(acl_set)
        entity_id = f"note_summary:{summary.note_type}:{summary.note_id}"

        # Source-of-truth bits used to seed related_entity_ids + the
        # chunk's chat/project breadcrumbs so Spotlight can deep-link
        # the result back into the right note URL.
        title = getattr(note, "title", "") or ""
        related_ids: list[str] = []
        chat_label: str | None = None
        chat_id_str: str | None = None
        thread_id_str: str | None = None
        project_id_str: str | None = None
        task_id_str: str | None = None
        note_id_str = str(summary.note_id)

        if summary.note_type == NOTE_TYPE_CHAT:
            chat_label = CHAT_TYPE_LABEL.get(getattr(note, "chat_type", 0))
            if chat_label and note.channel_id:
                chat_id_str = str(note.channel_id)
            if getattr(note, "thread_root_id", None):
                thread_id_str = str(note.thread_root_id)
            related_ids.append(f"note:chat:{summary.note_id}")
        elif summary.note_type == NOTE_TYPE_TASK:
            if getattr(note, "project_id", None):
                project_id_str = str(note.project_id)
            if getattr(note, "task_id", None):
                task_id_str = str(note.task_id)
            related_ids.append(f"note:task:{summary.note_id}")
        else:
            related_ids.append(f"note:personal:{summary.note_id}")

        title_for_display = title or f"{note_label.capitalize()} note #{summary.note_id}"
        owner_id_str = (
            str(getattr(note, "owner_id", "")) if getattr(note, "owner_id", None) else None
        )
        parent_id_str = (
            str(getattr(note, "parent_note_id", ""))
            if getattr(note, "parent_note_id", None)
            else None
        )
        chunk = Chunk(
            chunk_id=entity_id,
            entity_type="note_summary",
            entity_id=entity_id,
            chunk_type="note_summary",
            team_id=str(summary.team_id),
            acl_user_ids=acl_user_ids,
            title=f"Summary — {title_for_display}",
            search_text=summary.summary_text,
            snippet_text=make_snippet(summary.summary_text),
            related_entity_ids=related_ids,
            note_id=note_id_str,
            note_type=note_label,
            chat_type=chat_label,
            chat_id=chat_id_str,
            thread_id=thread_id_str,
            project_id=project_id_str,
            task_id=task_id_str,
            note_owner_id=owner_id_str,
            note_parent_id=parent_id_str,
            created_at=iso(summary.ts_created_at),
            updated_at=iso(summary.ts_updated_at),
        )
        yield EntityChunks(
            entity_type="note_summary",
            entity_id=entity_id,
            chunks=[chunk],
        )
