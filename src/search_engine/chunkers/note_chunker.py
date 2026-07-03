"""Note chunker for ChatNote / TaskNote / PersonalNote.

Phase 9 — heading-aware sections. A note's BlockNote body is split
into sections at each `type: "heading"` block; each section becomes
one chunk (`note_section`). Notes with no headings still produce a
single section chunk, equivalent to the old `note_title_body`
behavior. The note title is repeated in every section's
`search_text` so a query that mentions the note's overall topic can
still find the right section.

ACL is the union of:
  * the note owner,
  * the parent context's members (chat members for ChatNote, project
    members for TaskNote, just the owner for PersonalNote),
  * any explicit `NotePermissionMaster` grants on this note.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator, Optional

from origin.models.note.chat_note_models import ChatNoteMaster
from origin.models.note.common_note_models import NotePermissionMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.models.project.prj_models import ProjectMembers
from origin.search_engine.agent.acl import chat_acl_user_ids
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_LABEL,
    NOTE_TYPE_CHAT,
    NOTE_TYPE_PERSONAL,
    NOTE_TYPE_TASK,
    Chunk,
    EntityChunks,
    chat_entity_id,
    iso,
    make_snippet,
)
from origin.search_engine.text_extraction import extract_sections

# ----------------------------- ChatNote -----------------------------


def iter_chat_note_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = ChatNoteMaster.objects.select_related("team", "owner")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)

    notes = list(qs)
    if not notes:
        return

    # Pre-load NotePermissionMaster grants for these note ids.
    grants_by_note = _load_grants(NOTE_TYPE_CHAT, [n.note_id for n in notes])

    # Pre-resolve chat ACLs in batches keyed by channel UUID.
    acl_by_chat = _resolve_chat_acls(notes)

    for note in notes:
        if not note.team_id:
            continue
        team_id = str(note.team_id)
        acl = set()
        if note.owner_id:
            acl.add(str(note.owner_id))
        acl.update(acl_by_chat.get(note.channel_id, []))
        acl.update(grants_by_note.get(note.note_id, []))

        related = []
        chat_label = CHAT_TYPE_LABEL.get(note.chat_type)
        # `thread_root_id` is the v3 thread-root Message UUID; null for
        # non-thread notes. We surface it (when present) so Spotlight has
        # the coordinates to build a `/workspace/notes/chat/.../thread/X/
        # note/Y` deep link. `related_entity_ids` mirrors the value with
        # the same shape used for chat entities.
        thread_id_str = str(note.thread_root_id) if note.thread_root_id else None
        if chat_label and note.channel_id:
            related.append(
                chat_entity_id(
                    chat_label,
                    note.channel_id,
                    note.thread_root_id if note.thread_root_id else None,
                )
            )
        if note.parent_note_id:
            related.append(f"note:chat:{note.parent_note_id}")

        chunks = _note_to_section_chunks(
            note_type_label="chat",
            note_id=note.note_id,
            team_id=team_id,
            acl_user_ids=sorted(acl),
            title=note.title or f"Chat note {note.note_id}",
            body=note.body,
            related=related,
            created_at=note.ts_created_at,
            updated_at=note.ts_updated_at,
            # Surface chat coordinates on the chunk so Spotlight can
            # build the proper /workspace/notes/chat/... URL.
            chat_type_label=chat_label,
            chat_id=str(note.channel_id) if note.channel_id else None,
            thread_id=thread_id_str,
            note_owner_id=str(note.owner_id) if note.owner_id else None,
            note_parent_id=(str(note.parent_note_id) if note.parent_note_id else None),
        )
        if chunks:
            yield EntityChunks(
                entity_type="note",
                entity_id=f"note:chat:{note.note_id}",
                chunks=chunks,
            )


def _resolve_chat_acls(notes: list[ChatNoteMaster]) -> dict:
    """Map `channel_id` (UUID) → list of user_ids allowed in that channel."""
    # Chat notes are keyed on the v3 `Channel` UUID; resolve membership
    # per distinct channel via the UUID-native `chat_acl_user_ids`
    # (DM/GM/MDM via `ChannelMember`; PM via the channel's `project_id`).
    out: dict = {}
    seen: dict = {}
    for n in notes:
        if not n.channel_id or not n.chat_type:
            continue
        if n.channel_id in seen:
            continue
        seen[n.channel_id] = True
        out[n.channel_id] = sorted(chat_acl_user_ids(n.chat_type, n.channel_id))
    return out


# ----------------------------- TaskNote -----------------------------


def iter_task_note_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = TaskNoteMaster.objects.select_related("team", "project", "task", "owner")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)
    notes = list(qs)
    if not notes:
        return

    grants_by_note = _load_grants(NOTE_TYPE_TASK, [n.note_id for n in notes])

    # Project ACLs.
    project_ids = {n.project_id for n in notes if n.project_id}
    members_by_project: dict[int, list[str]] = defaultdict(list)
    for row in ProjectMembers.objects.filter(project_id__in=project_ids).values(
        "project_id", "attendee_id"
    ):
        if row["attendee_id"]:
            members_by_project[row["project_id"]].append(str(row["attendee_id"]))

    for note in notes:
        if not note.team_id:
            continue
        team_id = str(note.team_id)
        acl = set(members_by_project.get(note.project_id, []))
        if note.owner_id:
            acl.add(str(note.owner_id))
        acl.update(grants_by_note.get(note.note_id, []))

        related = []
        if note.task_id:
            related.append(f"task:{note.task_id}")
        if note.parent_note_id:
            related.append(f"note:task:{note.parent_note_id}")

        chunks = _note_to_section_chunks(
            note_type_label="task",
            note_id=note.note_id,
            team_id=team_id,
            acl_user_ids=sorted(acl),
            title=note.title or f"Task note {note.note_id}",
            body=note.body,
            related=related,
            created_at=note.ts_created_at,
            updated_at=note.ts_updated_at,
            project_id=str(note.project_id) if note.project_id else None,
            # Surface task_id so Spotlight can deep-link the task note
            # without falling through to /workspace/notes.
            task_id=str(note.task_id) if note.task_id else None,
            note_owner_id=str(note.owner_id) if note.owner_id else None,
            note_parent_id=(str(note.parent_note_id) if note.parent_note_id else None),
        )
        if chunks:
            yield EntityChunks(
                entity_type="note",
                entity_id=f"note:task:{note.note_id}",
                chunks=chunks,
            )


# ----------------------------- PersonalNote -----------------------------


def iter_personal_note_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = PersonalNoteMaster.objects.select_related("team", "owner")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)
    notes = list(qs)
    if not notes:
        return

    grants_by_note = _load_grants(NOTE_TYPE_PERSONAL, [n.note_id for n in notes])

    for note in notes:
        if not note.team_id:
            continue
        team_id = str(note.team_id)
        acl = set()
        if note.owner_id:
            acl.add(str(note.owner_id))
        acl.update(grants_by_note.get(note.note_id, []))

        related = []
        if note.parent_note_id:
            related.append(f"note:personal:{note.parent_note_id}")

        chunks = _note_to_section_chunks(
            note_type_label="personal",
            note_id=note.note_id,
            team_id=team_id,
            acl_user_ids=sorted(acl),
            title=note.title or f"Personal note {note.note_id}",
            body=note.body,
            related=related,
            created_at=note.ts_created_at,
            updated_at=note.ts_updated_at,
            note_owner_id=str(note.owner_id) if note.owner_id else None,
            note_parent_id=(str(note.parent_note_id) if note.parent_note_id else None),
        )
        if chunks:
            yield EntityChunks(
                entity_type="note",
                entity_id=f"note:personal:{note.note_id}",
                chunks=chunks,
            )


# ----------------------------- helpers -----------------------------


def _load_grants(note_type_code: int, note_ids: list[int]) -> dict[int, list[str]]:
    """note_id → list of user_id strings with any role on that note."""
    grants: dict[int, list[str]] = defaultdict(list)
    if not note_ids:
        return grants
    for row in NotePermissionMaster.objects.filter(
        note_type=note_type_code, note_id__in=note_ids
    ).values("note_id", "user_id"):
        if row["user_id"]:
            grants[row["note_id"]].append(str(row["user_id"]))
    return grants


def _note_to_section_chunks(
    *,
    note_type_label: str,
    note_id: int,
    team_id: str,
    acl_user_ids: list[str],
    title: str,
    body,
    related: list[str],
    created_at,
    updated_at,
    project_id: Optional[str] = None,
    # Task-note specific — populated for task notes so Spotlight can
    # build `/workspace/notes/task/project/.../task/<id>/note/<id>`.
    task_id: Optional[str] = None,
    # Chat-note specifics — populated when the note is attached to a
    # chat / thread so Spotlight can deep-link the result row to the
    # right `/workspace/notes/chat/...` URL. None for personal/task
    # notes, and None for chat notes attached to a chat without a
    # thread (the chat-note URL pattern requires a thread_id).
    chat_type_label: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    # v2 — note overlays.
    note_owner_id: Optional[str] = None,
    note_parent_id: Optional[str] = None,
) -> list[Chunk]:
    """Split the body into heading-bounded sections; one Chunk per section.

    The note title is included in each section's `search_text` so a
    query matching the note's overall topic surfaces the right
    section. The snippet stays section-local (heading + body start)
    so the UI shows what was actually matched.
    """
    sections = extract_sections(body)
    title_clean = (title or "").strip()

    # No body and no title → nothing to index.
    if not sections and not title_clean:
        return []

    # Title-only note: index just the title as one degenerate section.
    if not sections:
        sections = [("", "")]

    entity_id = f"note:{note_type_label}:{note_id}"
    out: list[Chunk] = []
    for idx, (heading, section_body) in enumerate(sections):
        # `search_text` includes the note title in every section so a
        # heading-only query like "Risks" still pulls the right note
        # by topical context.
        parts: list[str] = []
        if title_clean:
            parts.append(title_clean)
        if heading:
            parts.append(heading)
        if section_body:
            parts.append(section_body)
        combined = "\n".join(parts).strip()
        if not combined:
            continue

        # The snippet shows the section the user actually matched:
        # `Heading — body...`. Falls back to body or heading alone.
        if heading and section_body:
            snippet_source = f"{heading} — {section_body}"
        else:
            snippet_source = section_body or heading

        out.append(
            Chunk(
                chunk_id=f"{entity_id}:section:{idx}",
                entity_type="note",
                entity_id=entity_id,
                chunk_type="note_section",
                team_id=team_id,
                acl_user_ids=acl_user_ids,
                title=title,
                search_text=combined,
                snippet_text=make_snippet(snippet_source),
                note_id=str(note_id),
                note_type=note_type_label,
                project_id=project_id,
                task_id=task_id,
                chat_type=chat_type_label,
                chat_id=chat_id,
                thread_id=thread_id,
                related_entity_ids=related,
                note_owner_id=note_owner_id,
                note_parent_id=note_parent_id,
                created_at=iso(created_at),
                updated_at=iso(updated_at),
            )
        )
    return out


# ----------------------------- entry point -----------------------------


def iter_all_note_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from iter_chat_note_chunks(since)
    yield from iter_task_note_chunks(since)
    yield from iter_personal_note_chunks(since)
