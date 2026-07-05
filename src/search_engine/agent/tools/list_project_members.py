"""`list_project_members` tool — who is on ONE project (A1-ext, membership).

The membership graph (`ProjectMembers`) existed only as an ACL guard
inside other tools; nothing let the agent ANSWER "who is on the Website
Redesign project?" — `get_team_members` lists the whole team, which is
the wrong grain. This tool exposes the project-level roster.

ACL contract (same shape as `get_project_summary`):
  * Tenant guard: project.team_id must equal ctx.team_id.
  * Membership guard: ctx.user_id must itself appear in ProjectMembers
    for the requested project — a non-member may not enumerate another
    project's roster (it leaks organisational structure).
  * System / soft-deleted users are excluded from the result.
"""

from __future__ import annotations

from typing import Any

from origin.models.project.prj_models import ProjectMaster, ProjectMembers
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_project_id = args.get("project_id")
    try:
        project_id = int(raw_project_id)
    except (TypeError, ValueError):
        raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")

    try:
        project = ProjectMaster.objects.get(project_id=project_id, is_deleted=False)
    except ProjectMaster.DoesNotExist:
        raise ToolError(f"Project {project_id} not found.")

    if str(getattr(project, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: project belongs to a different team.")

    memberships = ProjectMembers.objects.filter(project_id=project_id).select_related("attendee")
    if not any(str(m.attendee_id) == ctx.user_id for m in memberships):
        raise ToolError(
            f"Not authorized to access project {project_id}. "
            "You are not a member of that project."
        )

    members = []
    for m in memberships:
        u = m.attendee
        if u is None or u.is_deleted or u.is_system_user:
            continue
        members.append(
            {
                "user_id": str(u.id),
                "username": u.username or "",
                "email": u.email or "",
                "joined_at": m.ts_joined_at.isoformat() if m.ts_joined_at else None,
            }
        )

    return {
        "project_id": project_id,
        "project_name": project.project_name,
        "members": members,
        "__summary__": f"{len(members)} member(s) on {project.project_name}",
    }


LIST_PROJECT_MEMBERS = Tool(
    name="list_project_members",
    description=(
        "List the members of ONE specific project (user_id, username, "
        "email, joined_at). Use this for 'who is on project X?' / "
        "'who's working on the redesign?' questions — NOT get_team_members, "
        "which lists the whole team regardless of project. Requires the "
        "project_id (find it via list_projects first if you only have a "
        "name). You must be a member of the project yourself."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": "The project's integer id (from list_projects).",
            },
        },
        "required": ["project_id"],
    },
    run=_run,
)
