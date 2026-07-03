"""`get_milestone_assignee_counts` tool — per-user milestone workload.

Pairs with `get_workload_distribution` (task side): that tool counts
open/wip/pending tasks per assignee via the single `TaskMaster.assignee_id`
FK. Milestones use a multi-assignee join (`MilestoneAssignees`) so they
need their own counter. Answers "who owns the most milestones?".

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only milestones in projects where ctx.user_id is a
    ProjectMember. Explicit `project_id` validated against the set.
  * Exclude soft-deleted milestones and inactive/system users.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Case, Count, When

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneAssignees
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_LIMIT = 50
_OPEN_STATUSES = ["Open", "WIP", "Pending"]


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
                f"Not authorized to query project {project_id}. "
                "You are not a member of that project."
            )
        scoped_project_ids = {project_id}

    try:
        limit = int(args.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, _MAX_LIMIT))

    qs = (
        MilestoneAssignees.objects.filter(
            team_id=ctx.team_id,
            milestone__project_id__in=scoped_project_ids,
            milestone__is_deleted=False,
            user__is_deleted=False,
            user__is_system_user=False,
        )
        .values("user_id", "user__username")
        .annotate(
            milestone_count=Count("milestone_id", distinct=True),
            open_milestone_count=Count(
                Case(
                    When(
                        milestone__status__in=_OPEN_STATUSES,
                        then="milestone_id",
                    )
                ),
                distinct=True,
            ),
        )
        .order_by("-milestone_count", "-open_milestone_count", "user__username")[:limit]
    )

    assignees = [
        {
            "user_id": str(r["user_id"]),
            "username": r["user__username"] or "",
            "milestone_count": r["milestone_count"] or 0,
            "open_milestone_count": r["open_milestone_count"] or 0,
        }
        for r in qs
    ]

    if assignees:
        head = ", ".join(
            f"{a['username']} ({a['open_milestone_count']} open / "
            f"{a['milestone_count']} total)"
            for a in assignees[:3]
        )
        summary = f"Top milestone owners: {head}" + (" …" if len(assignees) > 3 else "")
    else:
        summary = "No milestone assignments in scope."

    return {
        "project_id": int(raw_project_id) if raw_project_id is not None else None,
        "assignees": assignees,
        "__summary__": summary,
    }


GET_MILESTONE_ASSIGNEE_COUNTS = Tool(
    name="get_milestone_assignee_counts",
    description=(
        "Per-user count of milestones they're assigned to via the multi-"
        "assignee join. Use for 'who owns the most milestones?' or "
        "'who's leading work on the roadmap?'. Pairs with "
        "`get_workload_distribution` (task side of 'who has the most work?'). "
        "Returns `milestone_count` (all statuses) plus `open_milestone_count` "
        "(Open + WIP + Pending only). Scoped to milestones in projects "
        "the current user is a member of; deleted milestones and system / "
        "deactivated users are excluded."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to milestones in one project. Omit to span all "
                    "accessible projects."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max users to return (1–{_MAX_LIMIT}). Default 20.",
            },
        },
        "required": [],
    },
    run=_run,
)
