"""`get_my_task_summary` tool — personal workload rollup.

The "me" counterpart to `get_team_task_summary` / `get_project_summary`.
Aggregates every task in the workspace where the caller is the assignee,
across every project the caller is a member of.

Mirrors the `myStats` calculation in
`genos-frontend/src/features/tasks/components/dashboard/TaskHomeContent.tsx`
(lines 476-516) so the agent's numbers reconcile with the dashboard's
"My Tasks" KPI strip exactly: `openCount`, `wipCount`, `pendingCount`,
`closedCount`, `overdueCount`, `dueThisWeekCount`, `completionPct`.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership scope: tasks are counted only when the task's project is
    one ctx.user_id is a member of. The assignee filter is the natural
    "me" filter; the project-membership intersection is defence-in-depth.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext

_STATUS_LABELS = ["Open", "WIP", "Pending", "Closed"]
_PRIORITY_LABELS = ["Minimal", "Low", "Normal", "High", "Critical"]
_CLOSED_STATUSES = ["Closed", "Deleted"]
_PER_PROJECT_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    member_projects = list(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", "project__project_name")
    )
    member_project_ids = {pid for pid, _ in member_projects}
    project_name_by_id: dict[int, str | None] = {pid: name for pid, name in member_projects}

    today = timezone.now().date()
    week_ahead = today + timedelta(days=7)

    # Exclude `status='Deleted'` (the enum value, not the `is_deleted`
    # tombstone) — deleted-status rows are not actionable work.
    base_qs = TaskMaster.objects.filter(
        team_id=ctx.team_id,
        assignee_id=ctx.user_id,
        project_id__in=member_project_ids,
        is_deleted=False,
        is_init_task=False,
    ).exclude(status="Deleted")

    annotations: dict[str, Any] = {
        "total_tasks": Count("task_id", distinct=True),
        "overdue_count": Count(
            "task_id",
            distinct=True,
            filter=Q(due_date__isnull=False, due_date__lt=today) & ~Q(status__in=_CLOSED_STATUSES),
        ),
        "due_this_week_count": Count(
            "task_id",
            distinct=True,
            filter=Q(due_date__isnull=False, due_date__gte=today, due_date__lte=week_ahead)
            & ~Q(status__in=_CLOSED_STATUSES),
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

    total = agg.get("total_tasks", 0) or 0
    closed_n = status_breakdown.get("Closed", 0)
    open_n = status_breakdown.get("Open", 0)
    wip_n = status_breakdown.get("WIP", 0)
    pending_n = status_breakdown.get("Pending", 0)
    active_n = open_n + wip_n + pending_n
    overdue_n = agg.get("overdue_count", 0) or 0
    due_this_week_n = agg.get("due_this_week_count", 0) or 0
    completion_pct = round((closed_n / total) * 100) if total > 0 else 0

    # Per-project subtotals. Single aggregate query bucketed by
    # project_id so the agent can name "the project I have most work in"
    # without a follow-up call.
    per_project_rows = (
        base_qs.values("project_id")
        .annotate(
            total=Count("task_id", distinct=True),
            open_n=Count("task_id", distinct=True, filter=Q(status="Open")),
            overdue=Count(
                "task_id",
                distinct=True,
                filter=Q(due_date__isnull=False, due_date__lt=today)
                & ~Q(status__in=_CLOSED_STATUSES),
            ),
        )
        .order_by("-total")[:_PER_PROJECT_LIMIT]
    )
    per_project = [
        {
            "project_id": r["project_id"],
            "project_name": project_name_by_id.get(r["project_id"]),
            "total": r["total"] or 0,
            "open": r["open_n"] or 0,
            "overdue": r["overdue"] or 0,
        }
        for r in per_project_rows
    ]

    return {
        "total": total,
        "active_count": active_n,
        "status_breakdown": status_breakdown,
        "priority_breakdown": priority_breakdown,
        "overdue_count": overdue_n,
        "due_this_week_count": due_this_week_n,
        "completion_pct": completion_pct,
        "per_project": per_project,
        "__summary__": (
            f"My workload: {total} task(s) — {open_n} open, {wip_n} WIP, "
            f"{pending_n} pending, {closed_n} closed; {overdue_n} overdue, "
            f"{due_this_week_n} due this week"
        ),
    }


GET_MY_TASK_SUMMARY = Tool(
    name="get_my_task_summary",
    description=(
        "Personal task rollup for the current user. Use for 'what kind "
        "of WIP tasks do I have?', 'how many open tasks do I have?', "
        "'what's my workload?', or any 'me/my/I' question about task "
        "counts. Returns total, status_breakdown (Open / WIP / Pending / "
        "Closed), priority_breakdown (Minimal / Low / Normal / High / "
        "Critical), overdue_count, due_this_week_count (rolling 7 days), "
        "completion_pct, and per_project (top 10 projects by my total "
        "task count). Numbers match the dashboard's 'My Tasks' KPI strip "
        "exactly. For the prioritised next-up list, follow up with "
        "`get_my_focus_tasks`."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
    run=_run,
)
