"""`update_task` write tool — Phase 11.

Partial-update one `TaskMaster` row. Any subset of `title`,
`content_text`, `status`, `priority`, `effort_level`, `due_date`
may be supplied; omitted fields are left as-is. The model should
fetch the task first (via `fetch_task`) so it can propose changes
that respect the existing state.

ACL: same set as `fetch_task` — project members + assignee +
reporter. The requesting user must be on that list to write.

Approval flow (Phase 7): like all write tools, `requires_approval=True`.
The user sees the proposed argument set in the Approve / Reject card
and decides whether to execute.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_VALID_STATUSES = {"Open", "WIP", "Pending", "Closed", "Deleted"}
# Canonical enums live in `frontend/.../taskMeta.ts`. Keep in sync —
# a value outside the set still saves but renders without chip colour.
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
    """Same shape `create_task` produces — id / props / styles /
    children on every block, plus the trailing blank sentinel a
    user-typed body always carries."""
    if not text:
        return []
    lines = text.split("\n")
    return [_paragraph(line) for line in lines] + [_paragraph("")]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_task_id = args.get("task_id")
    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        raise ToolError(f"`task_id` must be an integer (got {raw_task_id!r}).")

    try:
        task = TaskMaster.objects.get(task_id=task_id)
    except TaskMaster.DoesNotExist:
        raise ToolError(f"Task {task_id} not found.")
    if task.is_deleted:
        raise ToolError(f"Task {task_id} has been deleted.")

    # Tenant + ACL: identical to the read-side fetch_task. A user who
    # can't see the task can't edit it.
    if str(getattr(task, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: task is in a different team.")
    allowed = task_acl_user_ids(
        getattr(task, "project_id", None),
        getattr(task, "assignee_id", None),
        getattr(task, "reporter_id", None),
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to update task {task_id}.")

    update_fields: list[str] = []
    changed: list[str] = []

    # --- title ---
    if "title" in args and args["title"] is not None:
        new_title = (args.get("title") or "").strip()
        if not new_title:
            raise ToolError("`title` must be non-empty if provided.")
        if new_title != task.title:
            task.title = new_title
            update_fields.append("title")
            changed.append("title")

    # --- content_text → BlockNote shape ---
    if "content_text" in args and args["content_text"] is not None:
        new_content = _wrap_blocknote((args.get("content_text") or "").strip())
        if new_content != task.content:
            task.content = new_content
            update_fields.append("content")
            changed.append("content")

    # --- status ---
    if "status" in args and args["status"] is not None:
        new_status = args["status"]
        if new_status not in _VALID_STATUSES:
            raise ToolError(
                f"`status` must be one of {sorted(_VALID_STATUSES)} (got {new_status!r})."
            )
        if new_status != task.status:
            task.status = new_status
            update_fields.append("status")
            changed.append("status")

    # --- priority ---
    if "priority" in args and args["priority"] is not None:
        new_priority = args["priority"]
        if new_priority not in _VALID_PRIORITIES:
            raise ToolError(
                f"`priority` must be one of {sorted(_VALID_PRIORITIES)} (got {new_priority!r})."
            )
        if new_priority != task.priority:
            task.priority = new_priority
            update_fields.append("priority")
            changed.append("priority")

    # --- effort_level ---
    if "effort_level" in args and args["effort_level"] is not None:
        new_effort = args["effort_level"]
        if new_effort not in _VALID_EFFORTS:
            raise ToolError(
                f"`effort_level` must be one of {sorted(_VALID_EFFORTS)} (got {new_effort!r})."
            )
        if new_effort != task.effort_level:
            task.effort_level = new_effort
            update_fields.append("effort_level")
            changed.append("effort_level")

    # --- due_date (ISO 8601 date string, or empty string to clear) ---
    if "due_date" in args and args["due_date"] is not None:
        raw_due = args["due_date"]
        if raw_due == "":
            if task.due_date is not None:
                task.due_date = None
                update_fields.append("due_date")
                changed.append("due_date(cleared)")
        else:
            try:
                new_due = date.fromisoformat(str(raw_due))
            except (TypeError, ValueError):
                raise ToolError(
                    f"`due_date` must be an ISO 8601 date (got {raw_due!r}). Use '' to clear."
                )
            if new_due != task.due_date:
                task.due_date = new_due
                update_fields.append("due_date")
                changed.append("due_date")

    if not update_fields:
        return {
            "task_id": task_id,
            "changed_fields": [],
            "__summary__": f"No changes applied to task {task.display_id}.",
        }

    try:
        task.save(update_fields=update_fields)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to update task: {e}")

    return {
        "task_id": task_id,
        "changed_fields": changed,
        "status": task.status,
        "title": task.title,
        "__summary__": f"Updated task {task.display_id}: {', '.join(changed)}",
    }


UPDATE_TASK = Tool(
    name="update_task",
    description=(
        "Update one or more fields of an existing task. REQUIRES USER "
        "APPROVAL — the user sees your proposed changes and decides "
        "whether to execute. Required: task_id. Optional (omit fields "
        "you don't want to change): title, content_text, status, "
        "priority, effort_level, due_date. Pass `due_date: ''` to "
        "clear the due date. Status enum: Open, WIP, Pending, Closed, "
        "Deleted. Use `fetch_task` first to see the current state so "
        "you don't propose a no-op."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "task_id": {
                "type": "INTEGER",
                "description": "Numeric task id to update.",
            },
            "title": {
                "type": "STRING",
                "description": "New title (1 line). Omit to leave unchanged.",
            },
            "content_text": {
                "type": "STRING",
                "description": (
                    "Replace the task body with this plain text. Omit to leave "
                    "unchanged. Pass '' to clear the body."
                ),
            },
            "status": {
                "type": "STRING",
                "enum": ["Open", "WIP", "Pending", "Closed", "Deleted"],
                "description": "New status. Omit to leave unchanged.",
            },
            "priority": {
                "type": "STRING",
                "enum": ["Minimal", "Low", "Normal", "High", "Critical"],
                "description": "New priority. Omit to leave unchanged.",
            },
            "effort_level": {
                "type": "STRING",
                "enum": ["Minimal", "Low", "Moderate", "High", "Extensive"],
                "description": "New effort estimate. Omit to leave unchanged.",
            },
            "due_date": {
                "type": "STRING",
                "description": (
                    "New ISO 8601 date (e.g. 2026-06-30). Omit to leave "
                    "unchanged. Pass '' to clear."
                ),
            },
        },
        "required": ["task_id"],
    },
    run=_run,
    requires_approval=True,
)
