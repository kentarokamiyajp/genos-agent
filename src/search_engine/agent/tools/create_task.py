"""`create_task` write tool — the first write tool in the registry.

Creates a new `TaskMaster` row under the given project. ACL: the
requesting user must be a member of the target project (same check
the read-side `fetch_task` uses).

This is the FIRST tool with `requires_approval=True`. The agent
controller routes any call to a `requires_approval` tool through the
pause/resume protocol introduced in Phase 7:

  1. Model invokes `create_task(...)` with proposed args.
  2. Controller emits `tool_call_pending_approval` and stops.
  3. Frontend renders an Approve / Reject card.
  4. User decides. POST `/api/v2/agent/decide/` resumes the loop.
  5. On approve: this tool's `_run` is called with the persisted args
     and the row is actually created. On reject: the tool is not
     called; the model sees a `{"error": "user_rejected"}` response
     and explains to the user.

`content_text` is wrapped into the minimal BlockNote document shape
that the rest of the app already produces — a single paragraph block.
The text-extraction layer (`text_extraction.extract_text`) reads this
shape natively when the new task gets indexed by the chunker.
"""

from __future__ import annotations

import uuid
from typing import Any

from origin.models.project.prj_models import ProjectMaster
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_DEFAULT_STATUS = "Open"
# Canonical enum lives on the frontend in `taskMeta.ts`. Keep this in
# sync — both the priority and effort-level sets are read by chip
# colour lookups, and an out-of-set value renders without a colour.
_VALID_PRIORITIES = {"Minimal", "Low", "Normal", "High", "Critical"}
_VALID_EFFORTS = {"Minimal", "Low", "Moderate", "High", "Extensive"}


_PARA_PROPS = {
    "textColor": "default",
    "textAlignment": "left",
    "backgroundColor": "default",
}


def _paragraph(text: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "paragraph",
        "props": dict(_PARA_PROPS),
        "content": ([{"text": text, "type": "text", "styles": {}}] if text else []),
        "children": [],
    }


def _wrap_blocknote(text: str) -> list[dict[str, Any]]:
    """Wrap plain text into the same BlockNote shape a user-typed task
    body produces. One paragraph per line + a trailing blank sentinel,
    each block carrying `id` / `props` / `children` / inline `styles`
    so the saved row is byte-equivalent to a user-typed one.
    `extract_text(...)` still walks this for the chunker on reindex.
    """
    if not text:
        return []
    lines = text.split("\n")
    return [_paragraph(line) for line in lines] + [_paragraph("")]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    title = (args.get("title") or "").strip()
    if not title:
        raise ToolError("`title` is required.")

    raw_project_id = args.get("project_id")
    try:
        project_id = int(raw_project_id)
    except (TypeError, ValueError):
        raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")

    try:
        project = ProjectMaster.objects.get(project_id=project_id, is_deleted=False)
    except ProjectMaster.DoesNotExist:
        raise ToolError(f"Project {project_id} not found.")

    # Tenant guard: project must be in the requesting team.
    if str(getattr(project, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: project belongs to a different team.")

    # ACL: requesting user must be a project member. Reuses the same
    # set the read-side `fetch_task` enforces, so a user who can't see
    # the project's tasks can't create one in it either.
    allowed = task_acl_user_ids(
        project_id=project_id,
        assignee_id=None,
        reporter_id=None,
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to create tasks in project {project_id}.")

    content_text = (args.get("content_text") or "").strip()
    priority = args.get("priority")
    if priority is not None and priority not in _VALID_PRIORITIES:
        raise ToolError(
            f"`priority` must be one of {sorted(_VALID_PRIORITIES)} (got {priority!r})."
        )
    effort_level = args.get("effort_level")
    if effort_level is not None and effort_level not in _VALID_EFFORTS:
        raise ToolError(
            f"`effort_level` must be one of {sorted(_VALID_EFFORTS)} (got {effort_level!r})."
        )
    due_date = args.get("due_date")
    # Let the model pass an ISO date string; pass it straight through to
    # Django which will parse it. Garbage gets a ToolError below.
    create_kwargs: dict[str, Any] = {
        "team_id": ctx.team_id,
        "project_id": project_id,
        "reporter_id": ctx.user_id,
        "title": title,
        "status": _DEFAULT_STATUS,
        "content": _wrap_blocknote(content_text),
    }
    if priority:
        create_kwargs["priority"] = priority
    if effort_level:
        create_kwargs["effort_level"] = effort_level
    if due_date:
        create_kwargs["due_date"] = due_date

    try:
        task = TaskMaster.objects.create(**create_kwargs)
    except Exception as e:  # noqa: BLE001 — surface as ToolError for the model
        raise ToolError(f"Failed to create task: {e}")

    return {
        "task_id": task.task_id,
        "project_id": project_id,
        "title": task.title,
        "status": task.status,
        "__summary__": f"Created task {task.display_id}: {task.title}",
    }


CREATE_TASK = Tool(
    name="create_task",
    description=(
        "Create a new task in a project. REQUIRES USER APPROVAL — the "
        "user will see your proposed arguments and decide whether to "
        "execute. Use this when the user explicitly asks you to create / "
        "add / file a task. Required: title, project_id. Optional: "
        "content_text (plain text body), priority, effort_level, due_date "
        "(ISO 8601, e.g. 2026-06-30). Status is always created as 'Open' "
        "— users change it via the UI."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "Short task title (1 line).",
            },
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Numeric project id the task belongs to. Resolve this "
                    "with `search_knowledge_base` first if the user "
                    "doesn't name it explicitly."
                ),
            },
            "content_text": {
                "type": "STRING",
                "description": ("Optional task body / description in plain text."),
            },
            "priority": {
                "type": "STRING",
                "enum": ["Minimal", "Low", "Normal", "High", "Critical"],
                "description": "Optional priority.",
            },
            "effort_level": {
                "type": "STRING",
                "enum": ["Minimal", "Low", "Moderate", "High", "Extensive"],
                "description": "Optional effort estimate.",
            },
            "due_date": {
                "type": "STRING",
                "description": "Optional ISO 8601 date, e.g. 2026-06-30.",
            },
        },
        "required": ["title", "project_id"],
    },
    run=_run,
    requires_approval=True,
)
