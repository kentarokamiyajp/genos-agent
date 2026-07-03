"""`get_project_activity_ranking` tool — rank projects by a metric.

Returns accessible projects sorted by one of:
  * `task_count`       — total non-deleted tasks (excludes init/draft).
  * `open_task_count`  — non-deleted, non-Closed tasks.
  * `note_count`       — total TaskNoteMaster rows on the project.

Use for "which project has the most task-notes?", "rank my projects
by activity", or "which project has the most open work right now?".

`note_count` covers TaskNoteMaster only (project-scoped task notes);
PersonalNoteMaster and ChatNoteMaster are not project-scoped and are
out of scope for this ranking.

ACL contract:
  * Tenant guard: ctx.team_id.
  * Membership guard: only projects where ctx.user_id is a
    ProjectMember are ranked. Non-member projects are never returned,
    even with rank 0 — that would still leak project existence.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Count, Q

from origin.models.project.prj_models import ProjectMaster, ProjectMembers
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_VALID_METRICS = {"task_count", "open_task_count", "note_count"}
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    metric = args.get("metric")
    if metric not in _VALID_METRICS:
        raise ToolError(f"`metric` must be one of {sorted(_VALID_METRICS)} (got {metric!r}).")

    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    if not member_project_ids:
        return {
            "metric": metric,
            "limit": limit,
            "projects": [],
            "__summary__": (
                "No accessible projects — you are not a member of any project " "in this team."
            ),
        }

    projects = ProjectMaster.objects.filter(
        team_id=ctx.team_id,
        is_deleted=False,
        project_id__in=member_project_ids,
    )

    if metric == "task_count":
        qs = projects.annotate(
            count=Count(
                "project_tasks_master",
                filter=Q(
                    project_tasks_master__is_deleted=False,
                    project_tasks_master__is_init_task=False,
                ),
            )
        )
    elif metric == "open_task_count":
        qs = projects.annotate(
            count=Count(
                "project_tasks_master",
                filter=Q(
                    project_tasks_master__is_deleted=False,
                    project_tasks_master__is_init_task=False,
                )
                & ~Q(project_tasks_master__status__in=["Closed", "Deleted"]),
            )
        )
    else:  # note_count
        qs = projects.annotate(count=Count("project_task_notes"))

    rows = qs.order_by("-count", "project_name")[:limit]

    ranked = [
        {
            "project_id": p.project_id,
            "project_name": p.project_name,
            "count": p.count,
        }
        for p in rows
    ]

    if ranked:
        head = ", ".join(f"{r['project_name']} ({r['count']})" for r in ranked[:3])
        summary = f"Top {len(ranked)} project(s) by {metric}: {head}" + (
            " …" if len(ranked) > 3 else ""
        )
    else:
        summary = f"No projects with {metric} data in scope."

    return {
        "metric": metric,
        "limit": limit,
        "projects": ranked,
        "__summary__": summary,
    }


GET_PROJECT_ACTIVITY_RANKING = Tool(
    name="get_project_activity_ranking",
    description=(
        "Rank accessible projects by one of three metrics: "
        "`task_count` (total non-deleted tasks), `open_task_count` "
        "(non-Closed tasks), or `note_count` (task notes; "
        "TaskNoteMaster only — personal and chat notes excluded). "
        "Use for 'which project has the most task notes?', 'rank my "
        "projects by activity', or 'which project has the most open "
        "work?'. Only projects the current user is a member of are "
        "returned. Ties broken alphabetically by project_name."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "metric": {
                "type": "STRING",
                "enum": ["task_count", "open_task_count", "note_count"],
                "description": (
                    "Dimension to rank by. `task_count` counts all "
                    "non-deleted tasks; `open_task_count` excludes "
                    "Closed; `note_count` covers TaskNoteMaster only."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max number of projects to return (1–{_MAX_LIMIT}). "
                    f"Default {_DEFAULT_LIMIT}."
                ),
            },
        },
        "required": ["metric"],
    },
    run=_run,
)
