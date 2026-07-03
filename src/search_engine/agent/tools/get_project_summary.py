"""`get_project_summary` tool — task-count statistics for one project.

Returns total task count, a breakdown by status, and an overdue count so
the model can answer "how is project X going?" or "how many open tasks
remain?" without fetching individual task records.

ACL contract:
  * Tenant guard: project.team_id must equal ctx.team_id.
  * Membership guard: ctx.user_id must appear in ProjectMembers for the
    requested project.  Statistics for a project the user isn't a member
    of are not returned — even though the counts themselves don't contain
    any user-authored text, enumerating task volumes for inaccessible
    projects would still leak organisational structure.
"""

from __future__ import annotations

from typing import Any

from django.utils import timezone

from origin.models.project.prj_models import ProjectMaster, ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_STATUS_LABELS = ["Open", "WIP", "Pending", "Closed"]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_project_id = args.get("project_id")
    try:
        project_id = int(raw_project_id)
    except (TypeError, ValueError):
        raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")

    # --- Fetch project and apply tenant + membership guards. ---
    try:
        project = ProjectMaster.objects.get(project_id=project_id, is_deleted=False)
    except ProjectMaster.DoesNotExist:
        raise ToolError(f"Project {project_id} not found.")

    if str(getattr(project, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: project belongs to a different team.")

    is_member = ProjectMembers.objects.filter(
        project_id=project_id,
        attendee_id=ctx.user_id,
    ).exists()
    if not is_member:
        raise ToolError(
            f"Not authorized to access project {project_id}. "
            "You are not a member of that project."
        )

    # --- Aggregate counts. ---
    qs = TaskMaster.objects.filter(
        team_id=ctx.team_id,
        project_id=project_id,
        is_deleted=False,
        is_init_task=False,
    )

    total = qs.count()

    status_breakdown = {s: qs.filter(status=s).count() for s in _STATUS_LABELS}

    today = timezone.now().date()
    overdue_count = (
        qs.exclude(status__in=["Closed", "Deleted"])
        .filter(due_date__isnull=False, due_date__lt=today)
        .count()
    )

    open_n = status_breakdown.get("Open", 0)
    wip_n = status_breakdown.get("WIP", 0)

    return {
        "project_id": project_id,
        "project_name": project.project_name,
        "total_tasks": total,
        "status_breakdown": status_breakdown,
        "overdue_count": overdue_count,
        "__summary__": (
            f"Project '{project.project_name}': {total} task(s) — "
            f"{open_n} open, {wip_n} WIP, {overdue_count} overdue"
        ),
    }


GET_PROJECT_SUMMARY = Tool(
    name="get_project_summary",
    description=(
        "Return task-count statistics for one project: total tasks, a "
        "breakdown by status (Open / WIP / Pending / Closed), and the "
        "number of overdue tasks. Use for 'how is project X going?', "
        "'how many open tasks remain?', or sprint overview questions. "
        "Resolve the project name to a numeric id with `list_projects` "
        "first if needed. Only accessible to project members."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": "Numeric project id to summarise.",
            },
        },
        "required": ["project_id"],
    },
    run=_run,
)
