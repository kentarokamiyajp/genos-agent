"""`assign_task` write tool — set or clear a task's assignee.

A dedicated tool (rather than reusing `update_task`) so:
  (a) the approval card shows "assign_task" with a clear
      assignee_username, not a raw UUID buried in update_task args;
  (b) the model has an unambiguous primitive for "assign this to me /
      to John / unassign" without having to compose a partial update.

ACL contract (three layers):
  1. Tenant guard: task.team_id must equal ctx.team_id.
  2. Editor guard: ctx.user_id must be in task_acl_user_ids(…) — the
     same set that `fetch_task` and `update_task` enforce.  A user who
     cannot read a task cannot assign it.
  3. Assignee guard: when assigning to another user, that user must be
     an active TeamMember of ctx.team_id.  This prevents cross-team
     assignments and prevents assigning to an arbitrary UUID supplied
     by the LLM that happens to be a valid user in another tenant.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.team_models import TeamMembers
from origin.models.common.user_models import CustomUser
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # --- Resolve and validate task_id ---
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

    # Layer 1 — tenant guard.
    if str(getattr(task, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: task is in a different team.")

    # Layer 2 — editor guard (same set as fetch_task / update_task).
    allowed = task_acl_user_ids(
        getattr(task, "project_id", None),
        getattr(task, "assignee_id", None),
        getattr(task, "reporter_id", None),
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to update task {task_id}.")

    # --- Resolve the new assignee (or None for unassign). ---
    raw_assignee_id = args.get("assignee_id")
    new_assignee_id: str | None = None
    new_assignee_username: str | None = None

    if raw_assignee_id and raw_assignee_id != "null":
        # Layer 3 — assignee must be an active member of THIS team.
        # This query is intentionally scoped to ctx.team_id so the LLM
        # cannot supply an arbitrary UUID from a different tenant.
        in_team = TeamMembers.objects.filter(
            team_id=ctx.team_id,
            attendee_id=raw_assignee_id,
            is_deleted=False,
        ).exists()
        if not in_team:
            raise ToolError(
                f"User {raw_assignee_id!r} is not an active member of this team. "
                "Use get_team_members to find valid user ids."
            )

        try:
            assignee = CustomUser.objects.get(id=raw_assignee_id)
        except (CustomUser.DoesNotExist, Exception):
            raise ToolError(f"User {raw_assignee_id!r} not found.")

        if assignee.is_deleted:
            raise ToolError(f"User {raw_assignee_id!r} account has been deleted.")

        new_assignee_id = str(assignee.id)
        new_assignee_username = assignee.username or str(assignee.id)

    # --- Apply the change. ---
    task.assignee_id = new_assignee_id
    try:
        task.save(update_fields=["assignee_id"])
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to assign task: {e}")

    if new_assignee_username:
        summary = f"Assigned task #{task_id} to {new_assignee_username}"
    else:
        summary = f"Unassigned task #{task_id}"

    return {
        "task_id": task_id,
        "assignee_id": new_assignee_id,
        "assignee_username": new_assignee_username,
        "__summary__": summary,
    }


ASSIGN_TASK = Tool(
    name="assign_task",
    description=(
        "Assign or unassign a task. REQUIRES USER APPROVAL — the user "
        "sees the proposed assignment before it is saved. "
        "To assign: pass task_id + assignee_id (a UUID from "
        "get_current_user or get_team_members). "
        "To unassign: omit assignee_id or pass null. "
        "The new assignee must be an active member of the current team. "
        "Use get_current_user first when the user says 'assign to me'."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "task_id": {
                "type": "INTEGER",
                "description": "Numeric task id to assign.",
            },
            "assignee_id": {
                "type": "STRING",
                "description": (
                    "UUID of the user to assign. Omit or pass null to "
                    "unassign. Resolve names to UUIDs with "
                    "get_team_members."
                ),
            },
        },
        "required": ["task_id"],
    },
    run=_run,
    requires_approval=True,
)
