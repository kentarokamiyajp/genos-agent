"""`get_team_task_summary` tool — cross-project task rollup.

The team-wide counterpart of `get_project_summary`. Aggregates tasks
across every project the requesting user is a member of so the agent
can answer "how many tasks are Open / WIP / Closed / Deleted across my
workspace?" without forcing the user to pick a project.

Differences from `get_project_summary`:
  * No `project_id` argument — scope is "all my projects".
  * `status_breakdown` is FIVE rows including "Deleted" (the enum
    value, distinct from the `is_deleted` tombstone). Soft-deleted
    rows are still excluded everywhere.
  * Adds `priority_breakdown` (Minimal/Low/Normal/High/Critical).
  * Adds a `per_project` list (top 10 by total task count) so the
    agent can spot which project drives the total without a second tool
    call.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership scope: every task counted lives in a project where
    ctx.user_id is a `ProjectMembers` row.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext

_STATUS_LABELS = ["Open", "WIP", "Pending", "Closed", "Deleted"]
_PRIORITY_LABELS = ["Minimal", "Low", "Normal", "High", "Critical"]
_CLOSED_STATUSES = ["Closed", "Deleted"]
_PER_PROJECT_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    member_projects = list(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", "project__project_name")
    )
    member_project_ids = {pid for pid, _ in member_projects}
    project_name_by_id: dict[int, str | None] = {pid: name for pid, name in member_projects}

    base_qs = TaskMaster.objects.filter(
        team_id=ctx.team_id,
        project_id__in=member_project_ids,
        is_deleted=False,
        is_init_task=False,
    )

    today = timezone.now().date()
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

    # Per-project subtotals. One aggregate query bucketed by project_id so
    # the agent can name "the noisiest project" without a second tool call.
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

    total = agg.get("total_tasks", 0) or 0
    open_n = status_breakdown.get("Open", 0)
    wip_n = status_breakdown.get("WIP", 0)
    overdue_n = agg.get("overdue_count", 0) or 0

    return {
        "team_id": ctx.team_id,
        "project_count": len(member_project_ids),
        "total_tasks": total,
        "status_breakdown": status_breakdown,
        "priority_breakdown": priority_breakdown,
        "overdue_count": overdue_n,
        "per_project": per_project,
        "__summary__": (
            f"Workspace: {total} task(s) across {len(member_project_ids)} "
            f"project(s) — {open_n} open, {wip_n} WIP, {overdue_n} overdue"
        ),
    }


GET_TEAM_TASK_SUMMARY = Tool(
    name="get_team_task_summary",
    description=(
        "Workspace-wide task rollup across every project the user is a "
        "member of. Use for 'how many tasks are Open / WIP / Closed / "
        "Deleted across my projects?' or 'what's the state of the workspace?' "
        "— questions that don't name a specific project. Returns "
        "total_tasks, status_breakdown (Open / WIP / Pending / Closed / "
        "Deleted), priority_breakdown (Minimal / Low / Normal / High / "
        "Critical), overdue_count, and per_project (top 10 projects by "
        "total task count). Use `get_project_summary(project_id)` when "
        "the user names a specific project — this tool is the multi-"
        "project counterpart."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
    run=_run,
)
