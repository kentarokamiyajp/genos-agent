"""`get_top_task_closers` tool — rank users by tasks closed.

Returns the top N users by count of TaskActivity rows of action_type
'closed' in the past N days. Use to answer "who is closing the most
tasks?" or recognise high-throughput contributors during retros.

ACL contract:
  * Tenant guard: scoped to ctx.team_id via `task__team_id`.
  * Membership guard: counts only span projects where ctx.user_id is a
    ProjectMember.
  * Explicit `project_id`: validated against membership.
  * Actor filter: deleted users and is_system_user accounts are
    excluded — same convention as `get_team_members`. Activities whose
    actor went null (SET_NULL on user delete) are skipped via
    `actor__isnull=False`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db.models import Count
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_activity_models import TaskActivity, TaskActivityActionType
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_DAYS = 365
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # --- Validate `days`. ---
    raw_days = args.get("days")
    try:
        days = int(raw_days)
    except (TypeError, ValueError):
        raise ToolError(f"`days` must be an integer (got {raw_days!r}).")
    if not 1 <= days <= _MAX_DAYS:
        raise ToolError(f"`days` must be between 1 and {_MAX_DAYS} (got {days}).")

    # --- Validate `limit`. ---
    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

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

    since = timezone.now() - timedelta(days=days)
    rows = (
        TaskActivity.objects.filter(
            ts_created_at__gte=since,
            action_type=TaskActivityActionType.CLOSED.value,
            task__team_id=ctx.team_id,
            task__project_id__in=scoped_project_ids,
            task__is_deleted=False,
            task__is_init_task=False,
            actor__isnull=False,
            actor__is_deleted=False,
            actor__is_system_user=False,
        )
        .values("actor_id", "actor__username")
        .annotate(closed_count=Count("activity_id"))
        .order_by("-closed_count", "actor__username")[:limit]
    )

    closers = [
        {
            "user_id": str(r["actor_id"]),
            "username": r["actor__username"] or "",
            "closed_count": r["closed_count"],
        }
        for r in rows
    ]

    if closers:
        head = ", ".join(f"{c['username']}: {c['closed_count']}" for c in closers[:3])
        summary = f"Top {len(closers)} closer(s) in past {days} day(s) — {head}" + (
            " …" if len(closers) > 3 else ""
        )
    else:
        summary = f"No tasks closed in past {days} day(s) within accessible projects."

    return {
        "days": days,
        "project_id": scoped_project_id,
        "limit": limit,
        "closers": closers,
        "__summary__": summary,
    }


GET_TOP_TASK_CLOSERS = Tool(
    name="get_top_task_closers",
    description=(
        "Rank team members by the number of tasks they closed in the past "
        "N days. Use for 'who is closing the most tasks?', 'top "
        "contributors this sprint', or to identify high-throughput "
        "engineers during retros. Returns a list of {user_id, username, "
        "closed_count} sorted by closed_count desc. Scoped to projects "
        "the current user is a member of; pass `project_id` to narrow."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "days": {
                "type": "INTEGER",
                "description": f"Window length in days (1–{_MAX_DAYS}).",
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
                    f"Max number of users to return (1–{_MAX_LIMIT}). "
                    f"Default {_DEFAULT_LIMIT}."
                ),
            },
        },
        "required": ["days"],
    },
    run=_run,
)
