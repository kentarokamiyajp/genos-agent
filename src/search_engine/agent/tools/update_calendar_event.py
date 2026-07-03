"""`update_calendar_event` write tool.

Partial-update an existing event. Any subset of summary, description,
location, start_iso, end_iso, time_zone, attendees, add_meet may be
supplied. Omitted fields are left as-is.

Quirks:
  - Re-scheduling requires BOTH `start_iso` and `end_iso` together.
    Google's PATCH replaces the entire start/end object, so a partial
    update creates an invalid event shape.
  - `attendees` is a full-replacement list, not a diff. Pass [] to
    clear all invitees.
  - `add_meet=true` attaches a Meet link to an event without one.
    `add_meet=false` removes any existing Meet link.

REQUIRES USER APPROVAL.
"""

from __future__ import annotations

import uuid
from typing import Any

from origin.search_engine.agent.calendar import (
    build_time_endpoint,
    calendar_api_error,
    emails_to_attendees,
    google_calendar_request,
    resolve_google_calendar_account,
    shape_event,
    validate_time_pair,
)
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    event_id = (args.get("event_id") or "").strip()
    if not event_id:
        raise ToolError("`event_id` is required.")
    calendar_id = args.get("calendar_id") or "primary"

    body: dict[str, Any] = {}
    params: dict[str, Any] = {}

    summary = args.get("summary")
    if isinstance(summary, str):
        s = summary.strip()
        if not s:
            raise ToolError("`summary` must be non-empty if provided.")
        body["summary"] = s

    description = args.get("description")
    if isinstance(description, str):
        body["description"] = description
    location = args.get("location")
    if isinstance(location, str):
        body["location"] = location

    start_iso = args.get("start_iso")
    end_iso = args.get("end_iso")
    if (start_iso is None) != (end_iso is None):
        raise ToolError(
            "Updating event timing requires BOTH `start_iso` and "
            "`end_iso` together — Google rejects half-updates."
        )
    if start_iso and end_iso:
        start_iso = start_iso.strip()
        end_iso = end_iso.strip()
        validate_time_pair(start_iso, end_iso)
        time_zone = args.get("time_zone") or None
        body["start"] = build_time_endpoint(start_iso, time_zone)
        body["end"] = build_time_endpoint(end_iso, time_zone)

    # Presence-check on attendees so the model can clear with []
    # (which is falsy and would be missed by a truthy check).
    if "attendees" in args:
        cleaned = emails_to_attendees(args.get("attendees"))
        body["attendees"] = cleaned
        if cleaned:
            params["sendUpdates"] = "none"

    if "add_meet" in args:
        if args.get("add_meet"):
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": uuid.uuid4().hex,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        else:
            body["conferenceData"] = None
        params["conferenceDataVersion"] = 1

    if not body:
        return {
            "event_id": event_id,
            "calendar_id": calendar_id,
            "changed_fields": [],
            "__summary__": f"No changes applied to event {event_id}.",
        }

    account = resolve_google_calendar_account(ctx.user_id)
    resp = google_calendar_request(
        account,
        "PATCH",
        f"/calendars/{calendar_id}/events/{event_id}",
        json=body,
        params=params or None,
    )
    if resp.status_code == 404:
        raise ToolError(
            f"Event {event_id} not found on calendar `{calendar_id}`. It "
            f"may have been deleted upstream."
        )
    if not resp.ok:
        raise calendar_api_error(resp, "update event")
    event = resp.json() or {}
    shaped = shape_event(event)
    return {
        **shaped,
        "calendar_id": calendar_id,
        "changed_fields": list(body.keys()),
        "__summary__": (f"Updated event {event_id}: {', '.join(body.keys())}."),
    }


UPDATE_CALENDAR_EVENT = Tool(
    name="update_calendar_event",
    description=(
        "Update one or more fields of an existing Google Calendar event. "
        "REQUIRES USER APPROVAL. Required: event_id. Optional (omit "
        "fields you don't want to change): summary, description, "
        "location, start_iso, end_iso, time_zone, attendees, add_meet, "
        "calendar_id. To re-schedule, supply BOTH start_iso and end_iso "
        "together (Google rejects half-updates of timing). `attendees` "
        "is a full-replacement list; pass [] to clear all invitees. "
        "`add_meet: false` removes an existing Meet link."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "event_id": {
                "type": "STRING",
                "description": "Google Calendar event id to update.",
            },
            "calendar_id": {
                "type": "STRING",
                "description": (
                    "Google calendar id the event lives on. Omit for " "the primary calendar."
                ),
            },
            "summary": {
                "type": "STRING",
                "description": "New event title.",
            },
            "start_iso": {
                "type": "STRING",
                "description": (
                    "New event start. Same shape rules as "
                    "create_calendar_event. Must be passed together "
                    "with `end_iso`."
                ),
            },
            "end_iso": {
                "type": "STRING",
                "description": (
                    "New event end. Same shape rules as "
                    "create_calendar_event. Must be passed together "
                    "with `start_iso`."
                ),
            },
            "time_zone": {
                "type": "STRING",
                "description": ("IANA timezone. Used only with timed start_iso/" "end_iso."),
            },
            "description": {
                "type": "STRING",
                "description": "New event description.",
            },
            "location": {
                "type": "STRING",
                "description": "New event location.",
            },
            "attendees": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": (
                    "Full replacement list of attendee emails. Pass [] " "to clear all invitees."
                ),
            },
            "add_meet": {
                "type": "BOOLEAN",
                "description": (
                    "true to attach a Google Meet link; false to " "remove an existing one."
                ),
            },
        },
        "required": ["event_id"],
    },
    run=_run,
    requires_approval=True,
)
