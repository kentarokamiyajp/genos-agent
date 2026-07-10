"""`update_note` write tool — edit an existing personal or task note.

Symmetric with `update_task`: partial-update title and/or body text of a
note the requesting user already has edit rights to.  Chat notes are out
of scope (same boundary as `create_note`).

ACL contract (UI parity — mirrors the REST layer's `require_write_role`):
  * Tenant guard: note.team_id must equal ctx.team_id.
  * WRITE access: the note owner always may edit; otherwise the user's
    effective role must be Editor or stronger (`get_effective_role`:
    explicit NotePermissionMaster row first, then implicit access —
    task-note project members are implicit Editors, personal notes grant
    no implicit access). An explicit Viewer row wins over the implicit
    fallback, so a deliberately-downgraded project member stays
    read-only. This matches exactly what the same user could do through
    the note UI.

Every successful title/body change also writes a version snapshot
(`snapshot_note_version`, same coalescing as the REST PUT) so agent
edits show up in the note's version history and can be restored.

All ids used for the ACL checks come from `ctx` or from the database
record itself — never from the LLM's function-call arguments.
"""

from __future__ import annotations

import logging
from typing import Any

from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.search_engine.agent.acl import owns_personal_folder
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.blocknote_md import markdown_to_blocks
from origin.search_engine.agent.tools.entity_links import resolve_note_entity_link
from origin.search_engine.chunkers.base import NOTE_TYPE_PERSONAL, NOTE_TYPE_TASK
from origin.views.utils.note_role import ROLE_EDITOR, get_effective_role
from origin.views.utils.note_version import snapshot_note_version

log = logging.getLogger(__name__)

_VALID_TYPES = {"personal", "task"}
_NOTE_TYPE_CODE = {"personal": NOTE_TYPE_PERSONAL, "task": NOTE_TYPE_TASK}


def _wrap_blocknote(text: str, ctx: ToolContext) -> list[dict[str, Any]]:
    """Parse the agent's markdown into structured BlockNote blocks so an
    updated note keeps headings / lists / emphasis instead of collapsing
    to one flat paragraph. Citation tokens (`[prose](task:12)`, bare
    `[task:12]`) become working in-app links, resolved team-scoped —
    see `blocknote_md` / `entity_links`."""
    return markdown_to_blocks(
        text,
        entity_link_resolver=lambda token: resolve_note_entity_link(
            token, team_id=ctx.team_id
        ),
    )


def _has_write_permission(
    *, owner_id, note_id: int, note_type_code: int, ctx: ToolContext
) -> bool:
    """True if ctx.user_id may edit the note.

    Check 1: owner — always has write permission (covers legacy personal
             notes created before role rows existed).
    Check 2: effective role is Editor or stronger — same resolution the
             REST layer's `require_write_role` uses: an explicit
             NotePermissionMaster row wins; otherwise task-note project
             members are implicit Editors (personal notes grant no
             implicit access).
    """
    if str(owner_id or "") == ctx.user_id:
        return True
    role = get_effective_role(ctx.user_id, note_type_code, note_id)
    return role is not None and role <= ROLE_EDITOR


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
    # `folder_id` present (incl. explicit null → unfile to top level) is a
    # move. Folders are personal-only.
    has_folder = "folder_id" in args
    if has_folder and note_type != "personal":
        raise ToolError("`folder_id` is only valid for personal notes.")
    if not has_title and not has_body and not has_folder:
        raise ToolError(
            "At least one of `title`, `content_text`, or `folder_id` must be provided."
        )

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

    # Optional folder move (personal-only). `folder_id` present in args —
    # including an explicit null — is a move; None unfiles to the top level.
    folder_provided = "folder_id" in args
    new_folder_id = None
    if folder_provided:
        # `folder_id` is only meaningful on ROOT notes; child notes ride
        # along with their root (see PersonalNoteMaster.folder_id).
        if getattr(note, "parent_note_id", None) is not None:
            raise ToolError(
                "Only top-level notes can be filed into a folder; this is a "
                "child note — move its parent (root) note instead."
            )
        raw = args.get("folder_id")
        if raw is not None and raw != "":
            try:
                new_folder_id = int(raw)
            except (TypeError, ValueError):
                raise ToolError(f"`folder_id` must be an integer or null (got {raw!r}).")
            if not owns_personal_folder(
                folder_id=new_folder_id, team_id=ctx.team_id, user_id=ctx.user_id
            ):
                raise ToolError(
                    f"Folder {new_folder_id} is not one of your personal-note folders. "
                    "Call list_note_folders to see the available folders and their ids."
                )

    return _apply_changes(
        note=note,
        args=args,
        ctx=ctx,
        has_title=has_title,
        has_body=has_body,
        note_id=note_id,
        note_type="personal",
        note_type_code=note_type_code,
        folder_provided=folder_provided,
        new_folder_id=new_folder_id,
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

    # Write ACL: owner, implicit project-member Editor, or explicit
    # editor grant — task notes are a shared surface within the project,
    # so members can edit them (matching the note UI). An explicit
    # Viewer row still locks a member out.
    if not _has_write_permission(
        owner_id=getattr(note, "owner_id", None),
        note_id=note_id,
        note_type_code=note_type_code,
        ctx=ctx,
    ):
        raise ToolError(
            f"Not authorized to edit task note {note_id}. "
            "You must be the note owner, a member of the note's project, "
            "or have an explicit editor grant."
        )

    return _apply_changes(
        note=note,
        args=args,
        ctx=ctx,
        has_title=has_title,
        has_body=has_body,
        note_id=note_id,
        note_type="task",
        note_type_code=note_type_code,
    )


def _note_parent_context(note, note_type: str) -> dict[str, Any] | None:
    """Task-note project/task refs in `fetch_note`'s parent_context shape
    so the controller's `_note_source` chip builder (and the frontend
    deep link) consume the result unchanged."""
    if note_type != "task":
        return None
    pc = {
        "project_id": str(note.project_id) if note.project_id else None,
        "task_id": str(note.task_id) if note.task_id else None,
    }
    return {k: v for k, v in pc.items() if v is not None}


def _apply_changes(
    *,
    note,
    args: dict[str, Any],
    ctx: ToolContext,
    has_title: bool,
    has_body: bool,
    note_id: int,
    note_type: str,
    note_type_code: int,
    folder_provided: bool = False,
    new_folder_id=None,
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
        new_body = _wrap_blocknote((args.get("content_text") or "").strip(), ctx)
        if new_body != (note.body or []):
            note.body = new_body
            update_fields.append("body")
            changed.append("body")

    # Folder move (personal-only; `new_folder_id` already validated as
    # owned, or None for top level).
    if folder_provided and getattr(note, "folder_id", None) != new_folder_id:
        note.folder_id = new_folder_id
        update_fields.append("folder_id")
        changed.append("folder")

    parent_context = _note_parent_context(note, note_type)

    if not update_fields:
        result = {
            "note_id": note_id,
            "note_type": note_type,
            "title": note.title,
            "changed_fields": [],
            "__summary__": f"No changes applied to {note_type} note #{note_id}.",
        }
        if parent_context:
            result["parent_context"] = parent_context
        return result

    try:
        note.save(update_fields=update_fields)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to update note: {e}")

    # Version snapshot for title/body changes (folder-only moves don't
    # snapshot — parity with the REST layer, whose move path is separate).
    # Post-save state, same coalescing as the REST PUT. Best-effort: a
    # snapshot failure must never fail the already-persisted edit.
    if {"title", "body"} & set(update_fields):
        try:
            from origin.models.common.user_models import CustomUser  # noqa: PLC0415

            snapshot_note_version(
                team=note.team,
                editor=CustomUser.objects.filter(id=ctx.user_id).first(),
                note_type=note_type_code,
                note_id=note_id,
                title=note.title,
                body=note.body,
            )
        except Exception:  # noqa: BLE001
            log.exception("update_note: version snapshot failed for note %s", note_id)

    result = {
        "note_id": note_id,
        "note_type": note_type,
        "title": note.title,
        "changed_fields": changed,
        "__summary__": f"Updated {note_type} note #{note_id}: {', '.join(changed)}",
    }
    if parent_context:
        result["parent_context"] = parent_context
    return result


UPDATE_NOTE = Tool(
    name="update_note",
    description=(
        "Update an existing note's title and/or body text, and/or move a "
        "personal note between sidebar folders. REQUIRES USER APPROVAL — the "
        "user sees your proposed changes before they are saved. "
        "Supports note_type 'personal' and 'task'. Chat notes are not supported. "
        "Edit rights mirror the note UI: the owner, anyone with an editor "
        "grant, and (for task notes) any member of the note's project. "
        "Use fetch_note first to read the current content before proposing "
        "content changes; use list_note_folders to resolve a folder name to "
        "its id before moving a note with folder_id."
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
                    "New body text. This REPLACES the whole body — pass the "
                    "FULL new body (everything the note should contain), "
                    "reproducing every section you are not changing, never "
                    "just the edited part. Omit to leave the body unchanged. "
                    "Pass '' to clear the body."
                ),
            },
            "folder_id": {
                "type": "INTEGER",
                "description": (
                    "Optional (only valid with note_type='personal'): move the "
                    "note into this My-Notes sidebar folder. Resolve the folder "
                    "NAME to its id with `list_note_folders` first. Pass null to "
                    "move it back to the top level. Omit to leave the folder "
                    "unchanged."
                ),
            },
        },
        "required": ["note_id", "note_type"],
    },
    run=_run,
    requires_approval=True,
)
