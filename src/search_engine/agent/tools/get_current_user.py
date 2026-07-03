"""`get_current_user` tool — return the requesting user's own identity.

This is the safest tool in the registry: it only ever returns data for
the user who made the request.  The id comes exclusively from `ctx.user_id`
(server-trusted, derived from the JWT at the view layer) — no argument
from the LLM can influence which user's data is returned.

Use this when the user says "assign this to me", "what notes have I
written?", or "am I a member of project X?" — situations where the model
needs a concrete UUID for the caller.
"""

from __future__ import annotations

from typing import Any

from origin.models.common.user_models import CustomUser
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    try:
        user = CustomUser.objects.get(id=ctx.user_id)
    except CustomUser.DoesNotExist:
        # Should never happen — ctx.user_id is validated at auth time.
        # Surface as ToolError so the model can explain gracefully.
        raise ToolError("Current user record not found.")

    return {
        "user_id": str(user.id),
        "username": user.username or "",
        "email": user.email or "",
        "__summary__": f"Current user: {user.username} <{user.email}>",
    }


GET_CURRENT_USER = Tool(
    name="get_current_user",
    description=(
        "Return the identity (user_id, username, email) of the user who is "
        "making the current request. Use this whenever the user says 'me', "
        "'myself', or 'I' and you need a concrete user_id — for example, "
        "before calling assign_task to assign a task to themselves. This "
        "tool never returns data for any other user."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
    run=_run,
)
