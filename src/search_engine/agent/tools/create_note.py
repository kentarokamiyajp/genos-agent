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

from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.models.note.task_note_models import TaskNoteMaster
from origin.models.project.prj_models import ProjectMaster
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_VALID_TYPES = {"personal", "task"}


def _wrap_blocknote(text: str) -> list[dict[str, Any]]:
    """Same shape `create_task` produces — keeps the chunker happy on reindex."""
    if not text:
        return []
    return [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]


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
    body = _wrap_blocknote(content_text)

    if note_type == "personal":
        return _create_personal(title=title, body=body, ctx=ctx)
    return _create_task_note(args=args, title=title, body=body, ctx=ctx)


def _create_personal(*, title: str, body, ctx: ToolContext) -> dict[str, Any]:
    """Personal notes have no ACL beyond ownership — the creator is the owner."""
    try:
        note = PersonalNoteMaster.objects.create(
            team_id=ctx.team_id,
            owner_id=ctx.user_id,
            title=title,
            body=body,
        )
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to create personal note: {e}")
    return {
        "note_id": note.note_id,
        "note_type": "personal",
        "title": note.title,
        "__summary__": f'Created personal note #{note.note_id}: "{title[:60]}"',
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

    try:
        note = TaskNoteMaster.objects.create(
            team_id=ctx.team_id,
            project_id=project_id,
            owner_id=ctx.user_id,
            task_id=task_id_field,
            title=title,
            body=body,
        )
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to create task note: {e}")

    return {
        "note_id": note.note_id,
        "note_type": "task",
        "project_id": project_id,
        "task_id": task_id_field,
        "title": note.title,
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
        "'personal' (private to the creator; no extra args) or 'task' "
        "(attached to a project, optionally to a specific task; needs "
        "project_id, optionally task_id). Chat-attached notes are not "
        "supported by this tool — users add those through the UI."
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
                "description": "Optional plain-text body. Omit for a title-only note.",
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
        },
        "required": ["note_type", "title"],
    },
    run=_run,
    requires_approval=True,
)
