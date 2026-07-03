"""`delete_calendar_event` write tool.

Delete an event by id. REQUIRES USER APPROVAL — destructive and not
easily reversed (the user has to recreate the event from scratch).

A 404 from Google is treated as success: the event was already gone,
so the user-visible state matches what they asked for.
"""

from __future__ import annotations

from typing import Any

from origin.search_engine.agent.calendar import (
    calendar_api_error,
    google_calendar_request,
    resolve_google_calendar_account,
)
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    event_id = (args.get("event_id") or "").strip()
    if not event_id:
        raise ToolError("`event_id` is required.")
    calendar_id = args.get("calendar_id") or "primary"

    account = resolve_google_calendar_account(ctx.user_id)
    resp = google_calendar_request(
        account, "DELETE", f"/calendars/{calendar_id}/events/{event_id}"
    )
    if resp.status_code == 404:
        return {
            "event_id": event_id,
            "calendar_id": calendar_id,
            "deleted": True,
            "already_gone": True,
            "__summary__": (f"Event {event_id} was already deleted on Google."),
        }
    if not resp.ok:
        raise calendar_api_error(resp, "delete event")
    return {
        "event_id": event_id,
        "calendar_id": calendar_id,
        "deleted": True,
        "already_gone": False,
        "__summary__": f"Deleted event {event_id}.",
    }


DELETE_CALENDAR_EVENT = Tool(
    name="delete_calendar_event",
    description=(
        "Delete a Google Calendar event by id. REQUIRES USER APPROVAL — "
        "destructive and not easily undone (the user would need to "
        "recreate the event manually). Use this when the user explicitly "
        "asks to cancel / remove / delete an event. Required: event_id. "
        "Optional: calendar_id (defaults to primary). A delete of an "
        "already-deleted event is treated as success."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "event_id": {
                "type": "STRING",
                "description": "Google Calendar event id to delete.",
            },
            "calendar_id": {
                "type": "STRING",
                "description": ("Google calendar id. Omit for the primary calendar."),
            },
        },
        "required": ["event_id"],
    },
    run=_run,
    requires_approval=True,
)
