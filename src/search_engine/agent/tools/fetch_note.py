"""`fetch_note` tool — load one note's full body text.

Dispatches by `note_type`:
  * "personal" → PersonalNoteMaster
  * "task"     → TaskNoteMaster
  * "chat"     → ChatNoteMaster

ACL: re-derives the same membership set the note-chunker stamped onto
the OpenSearch chunk at index time — owner + parent-context members
+ explicit NotePermissionMaster grants. See `agent.acl` for the
helpers.
"""

from __future__ import annotations

from typing import Any

from origin.models.note.chat_note_models import ChatNoteMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.search_engine.agent.acl import (
    chat_note_acl_user_ids,
    personal_note_acl_user_ids,
    task_note_acl_user_ids,
)
from origin.search_engine.agent.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    wrap_workspace_content,
)
from origin.search_engine.chunkers.base import CHAT_TYPE_LABEL
from origin.search_engine.text_extraction import extract_text

_VALID_TYPES = {"personal", "task", "chat"}


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    note_type = (args.get("note_type") or "").lower().strip()
    if note_type not in _VALID_TYPES:
        raise ToolError(
            f"Unknown note_type {note_type!r}; expected one of " f"{sorted(_VALID_TYPES)}."
        )

    raw_note_id = args.get("note_id")
    try:
        note_id = int(raw_note_id)
    except (TypeError, ValueError):
        raise ToolError(f"note_id must be an integer (got {raw_note_id!r}).")

    if note_type == "personal":
        return _fetch_personal(note_id, ctx)
    if note_type == "task":
        return _fetch_task_note(note_id, ctx)
    return _fetch_chat_note(note_id, ctx)


def _check_team(note_team_id, expected_team_id: str) -> None:
    if str(note_team_id or "") != expected_team_id:
        raise ToolError("Not authorized: note is in a different team.")


def _fetch_personal(note_id: int, ctx: ToolContext) -> dict[str, Any]:
    try:
        note = PersonalNoteMaster.objects.get(note_id=note_id)
    except PersonalNoteMaster.DoesNotExist:
        raise ToolError(f"Personal note {note_id} not found.")
    _check_team(note.team_id, ctx.team_id)

    allowed = personal_note_acl_user_ids(owner_id=getattr(note, "owner_id", None), note_id=note_id)
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to read personal note {note_id}.")

    return _shape_note(
        note_id=note_id,
        note_type="personal",
        title=note.title,
        body=note.body,
        parent_note_id=note.parent_note_id,
        ts_created_at=note.ts_created_at,
        ts_updated_at=note.ts_updated_at,
        parent_context={"owner_id": str(note.owner_id) if note.owner_id else None},
    )


def _fetch_task_note(note_id: int, ctx: ToolContext) -> dict[str, Any]:
    try:
        note = TaskNoteMaster.objects.get(note_id=note_id)
    except TaskNoteMaster.DoesNotExist:
        raise ToolError(f"Task note {note_id} not found.")
    _check_team(note.team_id, ctx.team_id)

    allowed = task_note_acl_user_ids(
        owner_id=getattr(note, "owner_id", None),
        project_id=getattr(note, "project_id", None),
        note_id=note_id,
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to read task note {note_id}.")

    return _shape_note(
        note_id=note_id,
        note_type="task",
        title=note.title,
        body=note.body,
        parent_note_id=note.parent_note_id,
        ts_created_at=note.ts_created_at,
        ts_updated_at=note.ts_updated_at,
        parent_context={
            "owner_id": str(note.owner_id) if note.owner_id else None,
            "project_id": str(note.project_id) if note.project_id else None,
            "task_id": str(note.task_id) if note.task_id else None,
        },
    )


def _fetch_chat_note(note_id: int, ctx: ToolContext) -> dict[str, Any]:
    try:
        note = ChatNoteMaster.objects.get(note_id=note_id)
    except ChatNoteMaster.DoesNotExist:
        raise ToolError(f"Chat note {note_id} not found.")
    _check_team(note.team_id, ctx.team_id)

    chat_type_code = note.chat_type
    channel_id = note.channel_id
    allowed = chat_note_acl_user_ids(
        owner_id=getattr(note, "owner_id", None),
        chat_type_code=chat_type_code,
        channel_id=channel_id,
        note_id=note_id,
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to read chat note {note_id}.")

    # parent_context KEY names chat_id / thread_id are opaque deep-link
    # feed-through; the source values are now the v3 channel / thread-root
    # UUID.
    return _shape_note(
        note_id=note_id,
        note_type="chat",
        title=note.title,
        body=note.body,
        parent_note_id=note.parent_note_id,
        ts_created_at=note.ts_created_at,
        ts_updated_at=note.ts_updated_at,
        parent_context={
            "owner_id": str(note.owner_id) if note.owner_id else None,
            "chat_type": CHAT_TYPE_LABEL.get(chat_type_code),
            "chat_id": str(channel_id) if channel_id else None,
            "is_thread": bool(note.is_thread),
            "thread_id": str(note.thread_root_id) if note.thread_root_id else None,
        },
    )


def _shape_note(
    *,
    note_id: int,
    note_type: str,
    title: str,
    body,
    parent_note_id,
    ts_created_at,
    ts_updated_at,
    parent_context: dict[str, Any],
) -> dict[str, Any]:
    body_text = extract_text(body)
    return {
        "note_id": note_id,
        "note_type": note_type,
        "title": title or "",
        "body_text": wrap_workspace_content(body_text),
        "parent_note_id": (str(parent_note_id) if parent_note_id else None),
        "parent_context": {k: v for k, v in parent_context.items() if v is not None},
        "ts_created": ts_created_at.isoformat() if ts_created_at else None,
        "ts_updated": ts_updated_at.isoformat() if ts_updated_at else None,
        "__summary__": (
            f"Loaded {note_type} note #{note_id}" + (f' "{title[:40]}"' if title else "")
        ),
    }


FETCH_NOTE = Tool(
    name="fetch_note",
    description=(
        "Load the full body text of one note. The system has three note "
        "types: personal (private), task (attached to a task), and chat "
        "(attached to a chat or thread). Use after `search_knowledge_base` "
        "when you need to read the entire note body, not just the snippet. "
        "ACL is enforced — only notes the user can access."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "note_type": {
                "type": "STRING",
                "enum": ["personal", "task", "chat"],
                "description": "Which note family the note_id refers to.",
            },
            "note_id": {
                "type": "INTEGER",
                "description": "Numeric note id.",
            },
        },
        "required": ["note_type", "note_id"],
    },
    run=_run,
)
