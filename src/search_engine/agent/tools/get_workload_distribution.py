"""`get_workload_distribution` tool — open task counts per assignee.

Returns, for each assignee in scope, the number of Open / WIP / Pending
tasks plus an `overdue_count` cross-cut (tasks with a past due_date
that are not Closed). Closed tasks are excluded — this is a *current*
workload snapshot, not a historical throughput report.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only tasks in projects where ctx.user_id is a
    ProjectMember are counted. Explicit `project_id` is validated
    against this set.
  * Unassigned tasks (assignee_id IS NULL) are excluded — they're a
    project-health signal, not a per-user workload measure.
  * Deleted/system users are excluded.

`overdue_count` is a SUBSET of the open+wip+pending counts, not a
separate bucket; the tool description tells the model this explicitly.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Case, Count, Q, When
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # --- Membership-scoped project ids (ACL). ---
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

    today = timezone.now().date()

    qs = (
        TaskMaster.objects.filter(
            team_id=ctx.team_id,
            project_id__in=scoped_project_ids,
            is_deleted=False,
            is_init_task=False,
            assignee_id__isnull=False,
            assignee__is_deleted=False,
            assignee__is_system_user=False,
        )
        .exclude(status__in=["Closed", "Deleted"])
        .values("assignee_id", "assignee__username")
        .annotate(
            open_count=Count(Case(When(status="Open", then=1))),
            wip_count=Count(Case(When(status="WIP", then=1))),
            pending_count=Count(Case(When(status="Pending", then=1))),
            overdue_count=Count(
                Case(
                    When(
                        Q(due_date__isnull=False)
                        & Q(due_date__lt=today)
                        & ~Q(status__in=["Closed", "Deleted"]),
                        then=1,
                    )
                )
            ),
        )
        .order_by("-open_count", "-wip_count", "assignee__username")
    )

    assignees = []
    for r in qs:
        active_total = (r["open_count"] or 0) + (r["wip_count"] or 0) + (r["pending_count"] or 0)
        # Skip noise: assignee rows where every conditional count is 0
        # (can happen if a status falls outside Open/WIP/Pending after
        # being excluded from Closed — e.g. a future status value).
        if active_total == 0 and (r["overdue_count"] or 0) == 0:
            continue
        assignees.append(
            {
                "user_id": str(r["assignee_id"]),
                "username": r["assignee__username"] or "",
                "open_count": r["open_count"] or 0,
                "wip_count": r["wip_count"] or 0,
                "pending_count": r["pending_count"] or 0,
                "overdue_count": r["overdue_count"] or 0,
            }
        )

    if assignees:
        head = ", ".join(
            f"{a['username']} ({a['open_count'] + a['wip_count'] + a['pending_count']} active, "
            f"{a['overdue_count']} overdue)"
            for a in assignees[:3]
        )
        summary = f"Workload across {len(assignees)} assignee(s): {head}" + (
            " …" if len(assignees) > 3 else ""
        )
    else:
        summary = "No open tasks across accessible projects."

    return {
        "project_id": scoped_project_id,
        "assignees": assignees,
        "__summary__": summary,
    }


GET_WORKLOAD_DISTRIBUTION = Tool(
    name="get_workload_distribution",
    description=(
        "Snapshot of current per-assignee workload: counts of Open, WIP, "
        "and Pending tasks per user, plus an `overdue_count` cross-cut "
        "(tasks past due_date that are not Closed). Use for 'who has the "
        "most open work?', 'is anyone overloaded?', or load-balancing "
        "questions. Note: `overdue_count` is a subset of open+wip+pending, "
        "NOT a separate bucket. Unassigned tasks are excluded. Closed "
        "tasks are excluded (use get_task_throughput_stats for historical "
        "throughput instead). Scoped to projects the current user is a "
        "member of."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to one project. Resolve the name to an id "
                    "with `list_projects` first if needed. Omit to span "
                    "all accessible projects."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
