"""`get_stale_tasks` tool — surface tasks that haven't moved in N days.

Returns non-Closed tasks whose `ts_updated_at` is older than `stale_days`
days ago, oldest first. Use to identify stuck work: tasks that haven't
been touched for a while and may need attention or cleanup.

This is the inverse of `get_task_throughput_stats` — throughput tells
you what is moving; this tells you what is not.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only tasks in projects where ctx.user_id is a
    ProjectMember.
  * Explicit `project_id`: validated against membership.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_DAYS = 365
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 20


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_days = args.get("stale_days")
    try:
        stale_days = int(raw_days)
    except (TypeError, ValueError):
        raise ToolError(f"`stale_days` must be an integer (got {raw_days!r}).")
    if not 1 <= stale_days <= _MAX_DAYS:
        raise ToolError(f"`stale_days` must be between 1 and {_MAX_DAYS} (got {stale_days}).")

    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    raw_project_id = args.get("project_id")
    scoped_project_id: int | None = None
    if raw_project_id is not None:
        try:
            scoped_project_id = int(raw_project_id)
        except (TypeError, ValueError):
            raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")
        if scoped_project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to query project {scoped_project_id}. "
                "You are not a member of that project."
            )
        scoped_project_ids = {scoped_project_id}
    else:
        scoped_project_ids = member_project_ids

    now = timezone.now()
    cutoff = now - timedelta(days=stale_days)

    qs = (
        TaskMaster.objects.filter(
            team_id=ctx.team_id,
            project_id__in=scoped_project_ids,
            is_deleted=False,
            is_init_task=False,
            ts_updated_at__lt=cutoff,
        )
        .exclude(status__in=["Closed", "Deleted"])
        .select_related("project")
        .order_by("ts_updated_at")[:limit]
    )

    tasks = []
    for t in qs:
        days_stale = (now - t.ts_updated_at).days
        tasks.append(
            {
                "task_id": t.task_id,
                "display_id": t.display_id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "assignee_id": str(t.assignee_id) if t.assignee_id else None,
                "project_id": t.project_id,
                "project_name": t.project.project_name if t.project else None,
                "days_stale": days_stale,
                "ts_updated_at": t.ts_updated_at.isoformat() if t.ts_updated_at else None,
            }
        )

    if tasks:
        summary = (
            f"Found {len(tasks)} task(s) unchanged for ≥{stale_days} day(s); "
            f"oldest is {tasks[0]['days_stale']} day(s) stale."
        )
    else:
        summary = f"No tasks unchanged for ≥{stale_days} day(s) in accessible projects."

    return {
        "stale_days": stale_days,
        "project_id": scoped_project_id,
        "tasks": tasks,
        "__summary__": summary,
    }


GET_STALE_TASKS = Tool(
    name="get_stale_tasks",
    description=(
        "Return non-Closed tasks that haven't been updated for at least "
        "`stale_days` days, oldest first. Use to identify stuck work — "
        "tasks that may need attention, reassignment, or cleanup. "
        "Returns task_id, title, status, priority, assignee_id, "
        "project_id, project_name, days_stale, and ts_updated_at. Scoped to projects "
        "the current user is a member of."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "stale_days": {
                "type": "INTEGER",
                "description": (
                    f"Minimum days since last update (1–{_MAX_DAYS}). "
                    "A task is included if ts_updated_at is older than "
                    "(now - stale_days)."
                ),
            },
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to one project. Resolve the name to an id "
                    "with `list_projects` first if needed."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max number of tasks to return (1–{_MAX_LIMIT}). "
                    f"Default {_DEFAULT_LIMIT}."
                ),
            },
        },
        "required": ["stale_days"],
    },
    run=_run,
)
