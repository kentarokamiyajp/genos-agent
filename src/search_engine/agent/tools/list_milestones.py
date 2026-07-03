"""`list_milestones` tool — milestone list with per-row task rollups.

Each milestone row carries `tasks_total`, `tasks_closed`, `overdue_count`,
and `assignee_count` — same aggregation source as the milestone table
view (`_serialize_milestone`). Answers questions like "how many tasks
does each milestone have?" without forcing the agent to call
`get_milestone_summary` once per milestone.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only milestones in projects where ctx.user_id is a
    ProjectMember. Explicit `project_id` validated against the set.
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

_MAX_LIMIT = 100
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

    raw_project_id = args.get("project_id")
    scoped_project_ids: set[int] = member_project_ids
    if raw_project_id is not None:
        try:
            project_id = int(raw_project_id)
        except (TypeError, ValueError):
            raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")
        if project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to list milestones in project {project_id}. "
                "You are not a member of that project."
            )
        scoped_project_ids = {project_id}

    raw_statuses = args.get("statuses")
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

    raw_assignee_user_id = args.get("assignee_user_id")
    assignee_user_id: str | None = None
    if raw_assignee_user_id:
        # We don't ACL-check the assignee id itself — it's a filter value,
        # not an auth claim. The caller's own project-membership scope on
        # `scoped_project_ids` still applies, so the filter only narrows
        # within milestones the caller could already see.
        assignee_user_id = str(raw_assignee_user_id)

    try:
        limit = int(args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, _MAX_LIMIT))

    qs = MilestoneMaster.objects.select_related("project", "task").filter(
        team_id=ctx.team_id,
        project_id__in=scoped_project_ids,
        is_deleted=False,
    )
    if statuses_filter:
        qs = qs.filter(status__in=statuses_filter)
    if assignee_user_id:
        # Join through the MilestoneAssignees table. `.distinct()` because
        # the join can match a milestone row twice if the user appears in
        # multiple assignee rows (the unique constraint prevents this, but
        # belt-and-braces matches the rest of the file's defensive style).
        qs = qs.filter(milestone_assignees__user_id=assignee_user_id).distinct()

    # Order: open work first, then by closest due_date, then most recent.
    # `due_date IS NULL` sorts last via Django's `nulls_last` (Postgres).
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
        assignee_count = m.milestone_assignees.count()
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
                "assignee_count": assignee_count,
            }
        )

    if milestones:
        head = ", ".join(
            f"{m['title']} ({m['tasks_closed']}/{m['tasks_total']} closed)" for m in milestones[:3]
        )
        summary = f"Found {len(milestones)} milestone(s): {head}" + (
            " …" if len(milestones) > 3 else ""
        )
    else:
        summary = "No milestones in scope."

    return {
        "project_id": int(raw_project_id) if raw_project_id is not None else None,
        "milestones": milestones,
        "__summary__": summary,
    }


LIST_MILESTONES = Tool(
    name="list_milestones",
    description=(
        "List milestones with per-row task rollups: tasks_total, "
        "tasks_closed, overdue_count, assignee_count. Use for 'how many "
        "tasks does each milestone have?', 'show me all milestones', "
        "'what milestones are in project X?'. Pair with "
        "`get_milestone_summary(milestone_id)` for a deeper per-milestone "
        "breakdown (status/priority/effort splits, assignee list). "
        "Counts include subtasks via the same Q-union as the milestone "
        "table view. Scoped to milestones in projects the current user "
        "is a member of."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to one project. Omit to span all accessible " "projects."
                ),
            },
            "statuses": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": ["Open", "WIP", "Pending", "Closed", "Deleted"],
                },
                "description": (
                    "Filter milestones by one or more status values. Omit "
                    "to include all non-tombstone milestones."
                ),
            },
            "assignee_user_id": {
                "type": "STRING",
                "description": (
                    "Filter to milestones where this user is an assignee "
                    "(via the MilestoneAssignees join). Use "
                    "`get_current_user` for the caller's own id, or "
                    "`get_team_members` to resolve a name to a UUID. "
                    "Prefer the dedicated `list_my_milestones` tool when "
                    "the question is about the caller themselves."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max milestones to return (1–{_MAX_LIMIT}). Default 50.",
            },
        },
        "required": [],
    },
    run=_run,
)
