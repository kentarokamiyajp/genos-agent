"""`get_sprint_summary` tool — per-sprint status / priority breakdown.

Returns total / per-status / per-priority task counts, overdue count,
days remaining until end_date, and the list of milestones associated
with the sprint. Answers "how is sprint X going?".

Counts source: only tasks with `sprint_id == sprint.sprint_id` are
included — sprint membership is a direct FK on TaskMaster, unlike
milestones which have the dual FK + parent_task_id linkage. Subtasks
inherit their parent's sprint when the parent is moved into a sprint
(see TaskMaster mutators in views/task/task_views.py), so a single
sprint_id filter is the canonical scope.

ACL contract:
  * Tenant guard: sprint.team_id == ctx.team_id.
  * Membership guard: sprint.project_id is in caller's ProjectMembers.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneMaster
from origin.models.task.sprint_models import Sprint
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.list_tasks import _milestone_task_q

_STATUS_LABELS = ["Open", "WIP", "Pending", "Closed", "Deleted"]
_PRIORITY_LABELS = ["Minimal", "Low", "Normal", "High", "Critical"]
_CLOSED_STATUSES = ["Closed", "Deleted"]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_id = args.get("sprint_id")
    try:
        sprint_id = int(raw_id)
    except (TypeError, ValueError):
        raise ToolError(f"`sprint_id` must be an integer (got {raw_id!r}).")

    try:
        sprint = Sprint.objects.select_related("project").get(
            sprint_id=sprint_id, is_deleted=False
        )
    except Sprint.DoesNotExist:
        raise ToolError(f"Sprint {sprint_id} not found.")

    if str(getattr(sprint, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: sprint is in a different team.")
    is_member = ProjectMembers.objects.filter(
        project_id=sprint.project_id,
        attendee_id=ctx.user_id,
    ).exists()
    if not is_member:
        raise ToolError(
            f"Not authorized to access sprint {sprint_id}. "
            "You are not a member of that project."
        )

    today = timezone.now().date()

    base_qs = TaskMaster.objects.filter(
        team_id=ctx.team_id,
        sprint_id=sprint_id,
        is_deleted=False,
        is_init_task=False,
    )

    annotations: dict[str, Any] = {
        "total_tasks": Count("task_id", distinct=True),
        "overdue_count": Count(
            "task_id",
            distinct=True,
            filter=Q(due_date__isnull=False, due_date__lt=today) & ~Q(status__in=_CLOSED_STATUSES),
        ),
    }
    for label in _STATUS_LABELS:
        annotations[f"_status_{label}"] = Count("task_id", distinct=True, filter=Q(status=label))
    for label in _PRIORITY_LABELS:
        annotations[f"_priority_{label}"] = Count(
            "task_id", distinct=True, filter=Q(priority=label)
        )
    agg = base_qs.aggregate(**annotations)

    status_breakdown = {label: agg.get(f"_status_{label}", 0) or 0 for label in _STATUS_LABELS}
    priority_breakdown = {
        label: agg.get(f"_priority_{label}", 0) or 0 for label in _PRIORITY_LABELS
    }

    # Milestone rollups for milestones associated with this sprint. Each
    # milestone uses the same Q-union as `_serialize_milestone` so the
    # counts reconcile with the UI's milestone table.
    milestones: list[dict[str, Any]] = []
    for m in MilestoneMaster.objects.select_related("task").filter(
        sprint_id=sprint_id,
        is_deleted=False,
    ):
        m_agg = (
            TaskMaster.objects.filter(_milestone_task_q(m))
            .distinct()
            .aggregate(
                tasks_total=Count("task_id", distinct=True),
                tasks_closed=Count(
                    "task_id", distinct=True, filter=Q(status__in=_CLOSED_STATUSES)
                ),
            )
        )
        milestones.append(
            {
                "milestone_id": m.milestone_id,
                "title": m.title,
                "status": m.status,
                "tasks_total": m_agg.get("tasks_total", 0) or 0,
                "tasks_closed": m_agg.get("tasks_closed", 0) or 0,
            }
        )

    # Signed days-until-end. Negative when the sprint has already ended
    # so the model can say "ran 3 days over" without separate plumbing.
    days_remaining: int | None = None
    if sprint.end_date is not None:
        days_remaining = (sprint.end_date - today).days

    total = agg.get("total_tasks", 0) or 0
    open_n = status_breakdown.get("Open", 0)
    wip_n = status_breakdown.get("WIP", 0)
    closed_n = status_breakdown.get("Closed", 0) + status_breakdown.get("Deleted", 0)
    overdue_n = agg.get("overdue_count", 0) or 0

    return {
        "sprint_id": sprint.sprint_id,
        "project_id": sprint.project_id,
        "project_name": sprint.project.project_name if sprint.project_id else None,
        "name": sprint.name,
        "sequence_number": sprint.sequence_number,
        "status": sprint.status,
        "start_date": sprint.start_date.isoformat() if sprint.start_date else None,
        "end_date": sprint.end_date.isoformat() if sprint.end_date else None,
        "days_remaining": days_remaining,
        "total_tasks": total,
        "status_breakdown": status_breakdown,
        "priority_breakdown": priority_breakdown,
        "overdue_count": overdue_n,
        "milestones": milestones,
        "__summary__": (
            f"Sprint '{sprint.name}' ({sprint.status}): {total} task(s) — "
            f"{open_n} open, {wip_n} WIP, {closed_n} closed, "
            f"{overdue_n} overdue"
            + (f", {days_remaining} day(s) left" if days_remaining is not None else "")
        ),
    }


GET_SPRINT_SUMMARY = Tool(
    name="get_sprint_summary",
    description=(
        "Per-sprint task rollup: total_tasks, status_breakdown (Open / "
        "WIP / Pending / Closed / Deleted), priority_breakdown (Minimal "
        "/ Low / Normal / High / Critical), overdue_count, "
        "`days_remaining` (signed: negative when the sprint has already "
        "ended), and a list of milestones in the sprint with their own "
        "tasks_total / tasks_closed. Use for 'how is the current sprint "
        "going?', 'what's overdue in sprint N?', or any sprint-progress "
        "question. Resolve a sprint name to an id with `list_sprints` "
        "first. Only accessible to project members."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "sprint_id": {
                "type": "INTEGER",
                "description": "Numeric sprint id to summarise.",
            },
        },
        "required": ["sprint_id"],
    },
    run=_run,
)
