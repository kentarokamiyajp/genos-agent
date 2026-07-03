"""`get_team_members` tool — list active users in the team.

Returns user_id, username, and email for every non-deleted, non-system
user who belongs to the requesting team.  This is the correct way for the
agent to resolve "assign to John" or "who is on this team?" — the model
gets a stable UUID → name mapping and can pass the UUID to `assign_task`.

ACL contract:
  * Only team members (ctx.team_id, via TeamMembers) are returned.
  * System users (is_system_user=True) are excluded — they are internal
    service accounts and should never appear in agent-visible lists.
  * Soft-deleted users (is_deleted=True) are excluded.
  * ctx.team_id is server-trusted; the tool takes no arguments that
    could influence which team's data is returned.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.team_models import TeamMembers
from origin.search_engine.agent.tools.base import Tool, ToolContext


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    memberships = (
        TeamMembers.objects.filter(
            team_id=ctx.team_id,
            is_deleted=False,
        )
        .select_related("attendee")
        .order_by("attendee__username")
    )

    members = []
    for m in memberships:
        u = m.attendee
        # Guard against orphaned membership rows and internal accounts.
        if u is None or u.is_deleted or u.is_system_user:
            continue
        members.append(
            {
                "user_id": str(u.id),
                "username": u.username or "",
                "email": u.email or "",
                "role": u.role or "",
            }
        )

    return {
        "members": members,
        "__summary__": f"{len(members)} active team member(s)",
    }


GET_TEAM_MEMBERS = Tool(
    name="get_team_members",
    description=(
        "List all active users in the team with their user_id, username, "
        "and email. Use this to resolve a person's name to their UUID "
        "before calling assign_task, or to answer 'who is on this team?' "
        "questions. Only returns real users in the current team — no "
        "system accounts, no deleted users, no cross-team data."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
    run=_run,
)
