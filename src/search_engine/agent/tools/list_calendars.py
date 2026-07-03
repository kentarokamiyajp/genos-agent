"""`list_calendars` tool — discover the user's Google calendars.

Returns id, summary, whether-primary, and color so the model can pick
a non-primary calendar in follow-up calls. Most users only have one
primary calendar; this tool's value is for users with separate work
and personal calendars who want to schedule on a specific one.
"""

from __future__ import annotations

from typing import Any

from origin.search_engine.agent.calendar import (
    calendar_api_error,
    google_calendar_request,
    resolve_google_calendar_account,
)
from origin.search_engine.agent.tools.base import Tool, ToolContext


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    account = resolve_google_calendar_account(ctx.user_id)
    resp = google_calendar_request(account, "GET", "/users/me/calendarList")
    if not resp.ok:
        raise calendar_api_error(resp, "list calendars")
    data = resp.json() or {}
    calendars = [
        {
            "id": c.get("id"),
            "summary": c.get("summary"),
            "primary": bool(c.get("primary", False)),
            "background_color": c.get("backgroundColor"),
            "access_role": c.get("accessRole"),
        }
        for c in data.get("items", [])
        if c.get("id")
    ]
    return {
        "calendars": calendars,
        "__summary__": f"Listed {len(calendars)} calendar(s).",
    }


LIST_CALENDARS = Tool(
    name="list_calendars",
    description=(
        "List the user's Google Calendars (id, summary, whether it's the "
        "primary calendar, access_role). Use this when the user mentions "
        "a non-primary calendar by name ('work', 'personal', a shared "
        "team calendar) and you need its id to pass as `calendar_id` to "
        "a follow-up event tool. For most requests you can skip this and "
        "default `calendar_id` to 'primary'. Requires the user to have "
        "connected Google via Integrations."
    ),
    parameters_schema={"type": "OBJECT", "properties": {}, "required": []},
    run=_run,
)
