"""`update_note` write tool — edit an existing personal or task note.

Symmetric with `update_task`: partial-update title and/or body text of a
note the requesting user already has edit rights to.  Chat notes are out
of scope (same boundary as `create_note`).

ACL contract (stricter than the read-side `fetch_note`):
  * Tenant guard: note.team_id must equal ctx.team_id.
  * READ access uses `personal_note_acl_user_ids` / `task_note_acl_user_ids`
    which admit owner + context members + explicit grants.
  * WRITE access is narrower:
      - The user must be the note owner (note.owner_id == ctx.user_id), OR
      - Have an explicit NotePermissionMaster row with role_id <= 2
        (1 = owner, 2 = editor).  Viewers (role_id = 3) cannot edit.
    For task notes, simply being a project member is NOT enough to write
    to someone else's note — an explicit editor grant is required.  This
    is the correct security posture: project membership gives read access
    to the note's content via search, not write access.

All ids used for the ACL checks come from `ctx` or from the database
record itself — never from the LLM's function-call arguments.
"""

from __future__ import annotations

from typing import Any

from origin.models.note.common_note_models import NotePermissionMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.chunkers.base import NOTE_TYPE_PERSONAL, NOTE_TYPE_TASK

_VALID_TYPES = {"personal", "task"}
_NOTE_TYPE_CODE = {"personal": NOTE_TYPE_PERSONAL, "task": NOTE_TYPE_TASK}


def _wrap_blocknote(text: str) -> list[dict[str, Any]]:
    """Same minimal BlockNote shape the other write tools produce."""
    if not text:
        return []
    return [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]


def _has_write_permission(
    *, owner_id, note_id: int, note_type_code: int, ctx: ToolContext
) -> bool:
    """True if ctx.user_id may edit the note.

    Check 1: owner — always has write permission.
    Check 2: explicit NotePermissionMaster row with role_id <= 2
             (owner=1, editor=2; viewer=3 is excluded).
    """
    if str(owner_id or "") == ctx.user_id:
        return True
    return NotePermissionMaster.objects.filter(
        user_id=ctx.user_id,
        note_id=note_id,
        note_type=note_type_code,
        role_id__lte=2,
    ).exists()


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    note_type = (args.get("note_type") or "").lower().strip()
    if note_type not in _VALID_TYPES:
        raise ToolError(
            f"`note_type` must be one of {sorted(_VALID_TYPES)} (got {note_type!r}). "
            "Chat notes are not supported by this tool."
        )

    raw_note_id = args.get("note_id")
    try:
        note_id = int(raw_note_id)
    except (TypeError, ValueError):
        raise ToolError(f"`note_id` must be an integer (got {raw_note_id!r}).")

    has_title = "title" in args and args["title"] is not None
    has_body = "content_text" in args and args["content_text"] is not None
    if not has_title and not has_body:
        raise ToolError("At least one of `title` or `content_text` must be provided.")

    note_type_code = _NOTE_TYPE_CODE[note_type]

    if note_type == "personal":
        return _update_personal(
            note_id=note_id,
            args=args,
            ctx=ctx,
            note_type_code=note_type_code,
            has_title=has_title,
            has_body=has_body,
        )
    return _update_task_note(
        note_id=note_id,
        args=args,
        ctx=ctx,
        note_type_code=note_type_code,
        has_title=has_title,
        has_body=has_body,
    )


def _update_personal(
    *,
    note_id: int,
    args: dict[str, Any],
    ctx: ToolContext,
    note_type_code: int,
    has_title: bool,
    has_body: bool,
) -> dict[str, Any]:
    try:
        note = PersonalNoteMaster.objects.get(note_id=note_id)
    except PersonalNoteMaster.DoesNotExist:
        raise ToolError(f"Personal note {note_id} not found.")

    # Tenant guard.
    if str(getattr(note, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: note belongs to a different team.")

    # Write ACL: owner or explicit editor.
    if not _has_write_permission(
        owner_id=getattr(note, "owner_id", None),
        note_id=note_id,
        note_type_code=note_type_code,
        ctx=ctx,
    ):
        raise ToolError(
            f"Not authorized to edit personal note {note_id}. "
            "You must be the owner or have explicit editor permission."
        )

    return _apply_changes(
        note=note,
        args=args,
        has_title=has_title,
        has_body=has_body,
        note_id=note_id,
        note_type="personal",
    )


def _update_task_note(
    *,
    note_id: int,
    args: dict[str, Any],
    ctx: ToolContext,
    note_type_code: int,
    has_title: bool,
    has_body: bool,
) -> dict[str, Any]:
    try:
        note = TaskNoteMaster.objects.get(note_id=note_id)
    except TaskNoteMaster.DoesNotExist:
        raise ToolError(f"Task note {note_id} not found.")

    # Tenant guard.
    if str(getattr(note, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: note belongs to a different team.")

    # Write ACL: owner or explicit editor.
    # Being a project member grants read access to task notes but NOT
    # write access — that requires an explicit NotePermissionMaster grant.
    if not _has_write_permission(
        owner_id=getattr(note, "owner_id", None),
        note_id=note_id,
        note_type_code=note_type_code,
        ctx=ctx,
    ):
        raise ToolError(
            f"Not authorized to edit task note {note_id}. "
            "You must be the note owner or have explicit editor permission."
        )

    return _apply_changes(
        note=note,
        args=args,
        has_title=has_title,
        has_body=has_body,
        note_id=note_id,
        note_type="task",
    )


def _apply_changes(
    *,
    note,
    args: dict[str, Any],
    has_title: bool,
    has_body: bool,
    note_id: int,
    note_type: str,
) -> dict[str, Any]:
    update_fields: list[str] = []
    changed: list[str] = []

    if has_title:
        new_title = (args.get("title") or "").strip()
        if not new_title:
            raise ToolError("`title` must be non-empty if provided.")
        if new_title != (note.title or ""):
            note.title = new_title
            update_fields.append("title")
            changed.append("title")

    if has_body:
        new_body = _wrap_blocknote((args.get("content_text") or "").strip())
        if new_body != (note.body or []):
            note.body = new_body
            update_fields.append("body")
            changed.append("body")

    if not update_fields:
        return {
            "note_id": note_id,
            "note_type": note_type,
            "changed_fields": [],
            "__summary__": f"No changes applied to {note_type} note #{note_id}.",
        }

    try:
        note.save(update_fields=update_fields)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to update note: {e}")

    return {
        "note_id": note_id,
        "note_type": note_type,
        "changed_fields": changed,
        "__summary__": f"Updated {note_type} note #{note_id}: {', '.join(changed)}",
    }


UPDATE_NOTE = Tool(
    name="update_note",
    description=(
        "Update an existing note's title and/or body text. REQUIRES USER "
        "APPROVAL — the user sees your proposed changes before they are saved. "
        "Supports note_type 'personal' and 'task'. Chat notes are not supported. "
        "You must be the note owner or have explicit editor permission — "
        "project membership alone is not sufficient. "
        "Use fetch_note first to read the current content before proposing changes."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "note_id": {
                "type": "INTEGER",
                "description": "Numeric note id to update.",
            },
            "note_type": {
                "type": "STRING",
                "enum": ["personal", "task"],
                "description": "Which note family the note_id refers to.",
            },
            "title": {
                "type": "STRING",
                "description": "New title (1 line). Omit to leave unchanged.",
            },
            "content_text": {
                "type": "STRING",
                "description": (
                    "New body text in plain text. Omit to leave unchanged. "
                    "Pass '' to clear the body."
                ),
            },
        },
        "required": ["note_id", "note_type"],
    },
    run=_run,
    requires_approval=True,
)
