"""`list_projects` tool — enumerate projects accessible to the requesting user.

ACL contract:
  * Only projects where the requesting user is listed in `ProjectMembers`
    are returned.  Membership is looked up by `ctx.user_id` (server-trusted),
    never by any id supplied in the LLM's function-call args.
  * The team_id guard ensures we never cross tenant boundaries even if a
    future change widens the ProjectMembers query.

Use this to resolve a human-readable project name to its numeric id before
calling `create_task`, `create_note`, or `get_project_summary`.
"""

from __future__ import annotations

from typing import Any

from origin.models.project.prj_models import ProjectMaster, ProjectMembers
from origin.search_engine.agent.tools.base import Tool, ToolContext


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # Derive accessible project ids from the membership table.
    # Double-scope: user must be a member AND the project must be in
    # the requesting team.  This prevents a user from seeing projects
    # they joined in a different team by guessing numeric ids.
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    if not member_project_ids:
        return {"projects": [], "__summary__": "No accessible projects found."}

    qs = ProjectMaster.objects.filter(
        team_id=ctx.team_id,
        is_deleted=False,
        project_id__in=member_project_ids,
    )

    name_filter = (args.get("name_filter") or "").strip()
    if name_filter:
        qs = qs.filter(project_name__icontains=name_filter)

    projects = []
    for p in qs.order_by("project_name"):
        member_count = ProjectMembers.objects.filter(project_id=p.project_id).count()
        projects.append(
            {
                "project_id": p.project_id,
                "project_name": p.project_name,
                "is_private": p.is_private,
                "member_count": member_count,
            }
        )

    summary = f"Found {len(projects)} accessible project(s)" + (
        f" matching '{name_filter}'" if name_filter else ""
    )
    return {"projects": projects, "__summary__": summary}


LIST_PROJECTS = Tool(
    name="list_projects",
    description=(
        "List all projects in the team that the current user is a member of. "
        "Use this to resolve a project name to its numeric project_id before "
        "calling create_task, create_note, or get_project_summary. Also useful "
        "for answering 'what projects do we have?' questions. Only returns "
        "projects the requesting user can access — no cross-user data leakage."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "name_filter": {
                "type": "STRING",
                "description": (
                    "Optional case-insensitive substring filter on project name. "
                    "Omit to list all accessible projects."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
