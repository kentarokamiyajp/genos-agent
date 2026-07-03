"""`get_milestone_summary` tool — task-count statistics for one milestone.

Returns total / per-status / per-priority / per-effort task counts and
overdue count, plus the milestone's assignees. Counts tasks via the
`_milestone_task_q` predicate from `list_tasks` so subtasks (linked
through `parent_task_id`) are included — same Q-union as
`_serialize_milestone` in milestone_views.py, so the agent rollup
reconciles with the table view.

Note on `status_breakdown`: it returns FIVE keys including "Deleted"
(the enum value, distinct from the soft-delete tombstone `is_deleted`).
Soft-deleted rows are still excluded everywhere. This is an explicit
divergence from `get_project_summary`'s four-key breakdown — the agent
needs the fifth bucket so it can answer the user's literal
'closed/opened/wip/deleted' question.

ACL contract:
  * Tenant guard: milestone.team_id must equal ctx.team_id.
  * Membership guard: milestone.project_id must be in caller's
    `ProjectMembers` set.
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

# Five-row breakdown (vs four in get_project_summary). "Deleted" is the
# enum value, not the is_deleted tombstone.
_STATUS_LABELS = ["Open", "WIP", "Pending", "Closed", "Deleted"]
_PRIORITY_LABELS = ["Minimal", "Low", "Normal", "High", "Critical"]
_EFFORT_LABELS = ["Minimal", "Low", "Moderate", "High", "Extensive"]
_CLOSED_STATUSES = ["Closed", "Deleted"]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_id = args.get("milestone_id")
    try:
        milestone_id = int(raw_id)
    except (TypeError, ValueError):
        raise ToolError(f"`milestone_id` must be an integer (got {raw_id!r}).")

    try:
        m = (
            MilestoneMaster.objects.select_related("project", "task")
            .prefetch_related("milestone_assignees__user")
            .get(milestone_id=milestone_id, is_deleted=False)
        )
    except MilestoneMaster.DoesNotExist:
        raise ToolError(f"Milestone {milestone_id} not found.")

    if str(getattr(m, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: milestone is in a different team.")

    is_member = ProjectMembers.objects.filter(
        project_id=m.project_id,
        attendee_id=ctx.user_id,
    ).exists()
    if not is_member:
        raise ToolError(
            f"Not authorized to access milestone {milestone_id}. "
            "You are not a member of that project."
        )

    today = timezone.now().date()

    # Single aggregate query for every breakdown so this is one DB round
    # trip rather than O(status × priority × effort) counts.
    qs = TaskMaster.objects.filter(_milestone_task_q(m)).distinct()
    annotations: dict[str, Any] = {
        "total_tasks": Count("task_id", distinct=True),
        "overdue_count": Count(
            "task_id",
            distinct=True,
            filter=Q(due_date__isnull=False, due_date__lt=today) & ~Q(status__in=_CLOSED_STATUSES),
        ),
        "unassigned_task_count": Count(
            "task_id", distinct=True, filter=Q(assignee_id__isnull=True)
        ),
    }
    for label in _STATUS_LABELS:
        annotations[f"_status_{label}"] = Count("task_id", distinct=True, filter=Q(status=label))
    for label in _PRIORITY_LABELS:
        annotations[f"_priority_{label}"] = Count(
            "task_id", distinct=True, filter=Q(priority=label)
        )
    for label in _EFFORT_LABELS:
        annotations[f"_effort_{label}"] = Count(
            "task_id", distinct=True, filter=Q(effort_level=label)
        )
    agg = qs.aggregate(**annotations)

    status_breakdown = {label: agg.get(f"_status_{label}", 0) or 0 for label in _STATUS_LABELS}
    priority_breakdown = {
        label: agg.get(f"_priority_{label}", 0) or 0 for label in _PRIORITY_LABELS
    }
    effort_breakdown = {label: agg.get(f"_effort_{label}", 0) or 0 for label in _EFFORT_LABELS}

    assignees = [
        {
            "user_id": str(a.user_id) if a.user_id else None,
            "username": getattr(a.user, "username", None) or "",
        }
        for a in m.milestone_assignees.all()
        if a.user_id is not None
    ]

    open_n = status_breakdown.get("Open", 0)
    wip_n = status_breakdown.get("WIP", 0)
    closed_n = status_breakdown.get("Closed", 0) + status_breakdown.get("Deleted", 0)
    overdue_n = agg.get("overdue_count", 0) or 0
    total = agg.get("total_tasks", 0) or 0

    return {
        "milestone_id": m.milestone_id,
        "project_id": m.project_id,
        "project_name": m.project.project_name if m.project_id else None,
        "title": m.title,
        "status": m.status,
        "priority": m.priority,
        "effort_level": m.effort_level,
        "due_date": m.due_date.isoformat() if m.due_date else None,
        "start_date": m.start_date.isoformat() if m.start_date else None,
        "assignees": assignees,
        "total_tasks": total,
        "unassigned_task_count": agg.get("unassigned_task_count", 0) or 0,
        "status_breakdown": status_breakdown,
        "priority_breakdown": priority_breakdown,
        "effort_breakdown": effort_breakdown,
        "overdue_count": overdue_n,
        "__summary__": (
            f"Milestone '{m.title}': {total} task(s) — "
            f"{open_n} open, {wip_n} WIP, {closed_n} closed, "
            f"{overdue_n} overdue"
        ),
    }


GET_MILESTONE_SUMMARY = Tool(
    name="get_milestone_summary",
    description=(
        "Return task-count statistics for one milestone: total tasks, "
        "status_breakdown (Open / WIP / Pending / Closed / Deleted — five "
        "rows including the Deleted status enum, NOT soft-delete "
        "tombstones), priority_breakdown (Minimal/Low/Normal/High/Critical), "
        "effort_breakdown (Minimal/Low/Moderate/High/Extensive), overdue "
        "count, unassigned count, and the milestone's assignees. Counts "
        "include subtasks (tasks whose parent_task_id is the milestone's "
        "backing task), matching the milestone table view. Use for 'how "
        "many tasks in milestone X?', 'how many Critical tasks under "
        "milestone X?', 'what's overdue in milestone X?'. Resolve a "
        "milestone name to a numeric id with `list_milestones` first if "
        "needed. Only accessible to project members."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "milestone_id": {
                "type": "INTEGER",
                "description": "Numeric milestone id to summarise.",
            },
        },
        "required": ["milestone_id"],
    },
    run=_run,
)
