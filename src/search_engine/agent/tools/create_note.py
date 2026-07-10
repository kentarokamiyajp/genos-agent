"""`create_note` write tool — Phase 11.

Create a new personal or task note. Chat-attached notes (which need
chat_type / chat_id / thread_id wiring) are deliberately out of scope
for the first cut — they have more failure modes and the agent rarely
needs to file one. Easy to add later.

Dispatch:
  * note_type='personal' → PersonalNoteMaster, owner = requesting user
  * note_type='task'     → TaskNoteMaster, attached to a project (and
                           optionally a specific task)

ACL:
  * personal: anyone — the note is private to the creator by definition
  * task: requesting user must be a project member (same set the
    read-side `fetch_note` enforces)

Approval flow (Phase 7): `requires_approval=True`. The user sees the
proposed title + body in the Approve / Reject card.
"""

from __future__ import annotations

from typing import Any

from django.db import transaction

from origin.models.note.common_note_models import NotePermissionMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.models.project.prj_models import ProjectMaster
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.acl import owns_personal_folder, task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.blocknote_md import markdown_to_blocks
from origin.search_engine.agent.tools.entity_links import resolve_note_entity_link
from origin.views.utils.note_role import NOTE_TYPE_PERSONAL, NOTE_TYPE_TASK, ROLE_OWNER
from origin.views.utils.note_version import snapshot_note_version

_VALID_TYPES = {"personal", "task"}


def _resolve_folder_id(args: dict[str, Any], note_type: str, ctx: ToolContext):
    """Validate an optional `folder_id` for a personal note. Returns the
    owned folder id (int) or None (top level). Folders are personal-only
    and owner-scoped — see `owns_personal_folder`."""
    raw = args.get("folder_id")
    if raw is None or raw == "":
        return None
    if note_type != "personal":
        raise ToolError("`folder_id` is only valid for personal notes.")
    try:
        folder_id = int(raw)
    except (TypeError, ValueError):
        raise ToolError(f"`folder_id` must be an integer (got {raw!r}).")
    if not owns_personal_folder(folder_id=folder_id, team_id=ctx.team_id, user_id=ctx.user_id):
        raise ToolError(
            f"Folder {folder_id} is not one of your personal-note folders. "
            "Call list_note_folders to see the available folders and their ids."
        )
    return folder_id


def _wrap_blocknote(text: str, ctx: ToolContext) -> list[dict[str, Any]]:
    """Parse the agent's markdown answer into structured BlockNote blocks
    (headings / lists / bold / italic) so the saved note keeps its
    formatting instead of being one flat paragraph. Citation tokens
    (`[prose](task:12)`, bare `[task:12]`) become working in-app links,
    resolved team-scoped — see `blocknote_md` / `entity_links`."""
    return markdown_to_blocks(
        text,
        entity_link_resolver=lambda token: resolve_note_entity_link(
            token, team_id=ctx.team_id
        ),
    )


def _snapshot_v1(*, note, note_type_code: int, ctx: ToolContext) -> None:
    """Initial version snapshot (v1) — parity with the REST create paths,
    so version history / restore works on agent-created notes from the
    first edit. Called inside the create transaction; the FK instances on
    the fresh note row spare extra lookups."""
    from origin.models.common.user_models import CustomUser  # noqa: PLC0415

    snapshot_note_version(
        team=note.team,
        editor=CustomUser.objects.filter(id=ctx.user_id).first(),
        note_type=note_type_code,
        note_id=note.note_id,
        title=note.title,
        body=note.body,
    )


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    note_type = (args.get("note_type") or "").lower().strip()
    if note_type not in _VALID_TYPES:
        raise ToolError(
            f"`note_type` must be one of {sorted(_VALID_TYPES)} (got {note_type!r}). "
            "Chat-attached notes are not supported by this tool."
        )

    title = (args.get("title") or "").strip()
    if not title:
        raise ToolError("`title` is required.")

    content_text = (args.get("content_text") or "").strip()
    body = _wrap_blocknote(content_text, ctx)

    folder_id = _resolve_folder_id(args, note_type, ctx)

    if note_type == "personal":
        return _create_personal(title=title, body=body, folder_id=folder_id, ctx=ctx)
    return _create_task_note(args=args, title=title, body=body, ctx=ctx)


def _create_personal(*, title: str, body, folder_id, ctx: ToolContext) -> dict[str, Any]:
    """Personal notes are gated by an explicit NotePermissionMaster role
    row, NOT by the `owner` column — `note_role.get_effective_role` grants
    personal notes *no* implicit access, so the read/write ACL checks
    (`require_read_role` / `require_write_role`) only pass when a role row
    exists. The UI create path (`PersonalNoteMasterView.post`) writes the
    note AND a ROLE_OWNER row in one transaction; this tool must do the
    same or the creator gets a 403 opening the note it just made.

    `folder_id` (already validated as owned, or None for top level) files
    the new root note into a sidebar folder.

    Wrapped in a transaction so a failure writing the role can't leave an
    orphaned, permanently-inaccessible note behind."""
    try:
        with transaction.atomic():
            note = PersonalNoteMaster.objects.create(
                team_id=ctx.team_id,
                owner_id=ctx.user_id,
                title=title,
                body=body,
                folder_id=folder_id,
            )
            NotePermissionMaster.objects.create(
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                note_id=note.note_id,
                note_type=NOTE_TYPE_PERSONAL,
                role_id=ROLE_OWNER,
            )
            _snapshot_v1(note=note, note_type_code=NOTE_TYPE_PERSONAL, ctx=ctx)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to create personal note: {e}")
    return {
        "note_id": note.note_id,
        "note_type": "personal",
        "title": note.title,
        "folder_id": folder_id,
        "__summary__": (
            f'Created personal note #{note.note_id}: "{title[:60]}"'
            + (f" in folder #{folder_id}" if folder_id else "")
        ),
    }


def _create_task_note(
    *, args: dict[str, Any], title: str, body, ctx: ToolContext
) -> dict[str, Any]:
    """Task notes require a project_id; an optional task_id attaches the
    note to a specific task within that project."""
    raw_project_id = args.get("project_id")
    try:
        project_id = int(raw_project_id)
    except (TypeError, ValueError):
        raise ToolError(f"`project_id` is required for note_type='task' (got {raw_project_id!r}).")

    try:
        project = ProjectMaster.objects.get(project_id=project_id, is_deleted=False)
    except ProjectMaster.DoesNotExist:
        raise ToolError(f"Project {project_id} not found.")

    if str(getattr(project, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: project belongs to a different team.")

    allowed = task_acl_user_ids(project_id=project_id, assignee_id=None, reporter_id=None)
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to create notes in project {project_id}.")

    # Optional task attachment. If supplied, verify the task lives in the
    # same project we just authorized against — prevents an oblique
    # write into an unrelated task.
    task_id_field = None
    raw_task_id = args.get("task_id")
    if raw_task_id is not None and raw_task_id != "":
        try:
            task_id_int = int(raw_task_id)
        except (TypeError, ValueError):
            raise ToolError(f"`task_id` must be an integer (got {raw_task_id!r}).")
        try:
            task = TaskMaster.objects.get(task_id=task_id_int, is_deleted=False)
        except TaskMaster.DoesNotExist:
            raise ToolError(f"Task {task_id_int} not found.")
        if task.project_id != project_id:
            raise ToolError(
                f"Task {task_id_int} belongs to project {task.project_id}, "
                f"not project {project_id}."
            )
        task_id_field = task_id_int

    # Transaction: note + explicit ROLE_OWNER row + v1 snapshot, same
    # trio the REST create path writes. The owner row is redundant for
    # ACL (project members are implicit Editors) but keeps the note's
    # members list and owner-gated actions consistent with UI-created
    # task notes.
    try:
        with transaction.atomic():
            note = TaskNoteMaster.objects.create(
                team_id=ctx.team_id,
                project_id=project_id,
                owner_id=ctx.user_id,
                task_id=task_id_field,
                title=title,
                body=body,
            )
            NotePermissionMaster.objects.create(
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                note_id=note.note_id,
                note_type=NOTE_TYPE_TASK,
                role_id=ROLE_OWNER,
            )
            _snapshot_v1(note=note, note_type_code=NOTE_TYPE_TASK, ctx=ctx)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to create task note: {e}")

    return {
        "note_id": note.note_id,
        "note_type": "task",
        "project_id": project_id,
        "task_id": task_id_field,
        "title": note.title,
        "parent_context": {
            k: v
            for k, v in {
                "project_id": str(project_id),
                "task_id": str(task_id_field) if task_id_field else None,
            }.items()
            if v is not None
        },
        "__summary__": (
            f'Created task note #{note.note_id}: "{title[:60]}"'
            + (f" (attached to task #{task_id_field})" if task_id_field else "")
        ),
    }


CREATE_NOTE = Tool(
    name="create_note",
    description=(
        "Create a new note. REQUIRES USER APPROVAL — the user sees your "
        "proposed title and body before it's saved. Two types: "
        "'personal' (private to the creator; optionally filed into a "
        "sidebar folder via folder_id) or 'task' (attached to a project, "
        "optionally to a specific task; needs project_id, optionally "
        "task_id). Use the task type for plan/research DOCUMENTS about a "
        "task or milestone (a milestone's note attaches to its backing "
        "task_id — see list_milestones); when the user wants TASKS "
        "created instead of a document, use create_task_plan. "
        "Chat-attached notes are not supported by this tool — "
        "users add those through the UI."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "note_type": {
                "type": "STRING",
                "enum": ["personal", "task"],
                "description": (
                    "'personal' for a private note, 'task' for a " "project / task-attached note."
                ),
            },
            "title": {
                "type": "STRING",
                "description": "Short note title (1 line).",
            },
            "content_text": {
                "type": "STRING",
                "description": (
                    "The note body as plain text. When the user asks to save "
                    "or file a previous answer, put the ENTIRE answer here "
                    "verbatim — every section, in full. Do NOT summarize, "
                    "abbreviate, or drop later sections; the note should "
                    "contain the whole thing, not a recap. Omit only for a "
                    "deliberately title-only note."
                ),
            },
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Required when note_type='task': the project the note "
                    "belongs to. Resolve with `search_knowledge_base` if "
                    "the user doesn't name it explicitly."
                ),
            },
            "task_id": {
                "type": "INTEGER",
                "description": (
                    "Optional (only valid with note_type='task'): attach "
                    "the note to a specific task within the project."
                ),
            },
            "folder_id": {
                "type": "INTEGER",
                "description": (
                    "Optional (only valid with note_type='personal'): file "
                    "the note into a My-Notes sidebar folder. Resolve the "
                    "folder NAME to its id with `list_note_folders` first. "
                    "Omit to leave the note at the top level."
                ),
            },
        },
        "required": ["note_type", "title"],
    },
    run=_run,
    requires_approval=True,
)
