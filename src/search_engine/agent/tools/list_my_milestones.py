"""`list_my_milestones` tool — milestones the caller is an assignee on.

The "me" counterpart to `list_milestones`. Always scoped to
`MilestoneAssignees.user_id == ctx.user_id`. Per-row rollups
(`tasks_total`, `tasks_closed`, `overdue_count`) match the same
`_milestone_task_q` source used by `list_milestones` / `get_milestone_summary`
so counts reconcile across tools.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only milestones in projects where ctx.user_id is a
    ProjectMember are included — defence-in-depth, even though the user
    must be a project member to have been added as an assignee in the
    first place.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneMaster
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.list_tasks import _milestone_task_q

_MAX_LIMIT = 50
_VALID_STATUSES = {"Open", "WIP", "Pending", "Closed", "Deleted"}
_CLOSED_STATUSES = ["Closed", "Deleted"]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    raw_statuses = args.get("status")
    statuses_filter: list[str] | None = None
    if raw_statuses is not None:
        if isinstance(raw_statuses, str):
            raw_statuses = [raw_statuses]
        invalid = set(raw_statuses) - _VALID_STATUSES
        if invalid:
            raise ToolError(
                f"Invalid status value(s): {sorted(invalid)}. "
                f"Must be one of {sorted(_VALID_STATUSES)}."
            )
        statuses_filter = list(raw_statuses)

    try:
        limit = int(args.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, _MAX_LIMIT))

    qs = (
        MilestoneMaster.objects.select_related("project", "task")
        .filter(
            team_id=ctx.team_id,
            project_id__in=member_project_ids,
            is_deleted=False,
            milestone_assignees__user_id=ctx.user_id,
        )
        .distinct()
    )
    if statuses_filter:
        qs = qs.filter(status__in=statuses_filter)

    qs = qs.order_by("status", "due_date", "-ts_updated_at")[:limit]

    today = timezone.now().date()
    milestones: list[dict[str, Any]] = []
    for m in qs:
        task_qs = TaskMaster.objects.filter(_milestone_task_q(m)).distinct()
        agg = task_qs.aggregate(
            tasks_total=Count("task_id", distinct=True),
            tasks_closed=Count("task_id", distinct=True, filter=Q(status__in=_CLOSED_STATUSES)),
            overdue=Count(
                "task_id",
                distinct=True,
                filter=Q(due_date__isnull=False, due_date__lt=today)
                & ~Q(status__in=_CLOSED_STATUSES),
            ),
        )
        milestones.append(
            {
                "milestone_id": m.milestone_id,
                "project_id": m.project_id,
                "project_name": m.project.project_name if m.project_id else None,
                "title": m.title,
                "status": m.status,
                "priority": m.priority,
                "due_date": m.due_date.isoformat() if m.due_date else None,
                "tasks_total": agg.get("tasks_total", 0) or 0,
                "tasks_closed": agg.get("tasks_closed", 0) or 0,
                "overdue_count": agg.get("overdue", 0) or 0,
            }
        )

    if milestones:
        head = ", ".join(
            f"{m['title']} ({m['tasks_closed']}/{m['tasks_total']} closed)" for m in milestones[:3]
        )
        summary = f"You are on {len(milestones)} milestone(s): {head}" + (
            " …" if len(milestones) > 3 else ""
        )
    else:
        summary = "You are not assigned to any milestones."

    return {
        "milestones": milestones,
        "__summary__": summary,
    }


LIST_MY_MILESTONES = Tool(
    name="list_my_milestones",
    description=(
        "List the milestones the current user is assigned to (via the "
        "MilestoneAssignees join). Use for 'what milestones am I on?', "
        "'what milestones am I responsible for?', or 'what milestones do "
        "I own?'. Returns per-row task rollups (tasks_total, "
        "tasks_closed, overdue_count) so the agent can summarise progress "
        "in one call. Pair with `get_milestone_summary(milestone_id)` for "
        "the deeper per-milestone breakdown. Caller is always the current "
        "user — there is no way to specify another user."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "status": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": ["Open", "WIP", "Pending", "Closed", "Deleted"],
                },
                "description": (
                    "Filter by one or more milestone statuses. Omit to "
                    "include all non-tombstone milestones."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max milestones to return (1–{_MAX_LIMIT}). Default 20.",
            },
        },
        "required": [],
    },
    run=_run,
)
