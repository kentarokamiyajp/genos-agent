"""`list_calendar_events` tool — list events in a date range.

Calls Google Calendar v3 `events.list` with `singleEvents=true` so
recurring events expand into individual instances. Results are sorted
by start time. Free-form text (summary, description) is wrapped via
`wrap_workspace_content` inside `shape_event`.
"""

from __future__ import annotations

from typing import Any

from origin.search_engine.agent.calendar import (
    calendar_api_error,
    google_calendar_request,
    resolve_google_calendar_account,
    shape_event,
)
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

# Upper bound on a single call; the model can paginate by adjusting
# `from`/`to` if it needs more.
_MAX_RESULTS_CAP = 100


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    account = resolve_google_calendar_account(ctx.user_id)
    calendar_id = args.get("calendar_id") or "primary"

    params: dict[str, Any] = {
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    time_min = args.get("from")
    time_max = args.get("to")
    if isinstance(time_min, str) and time_min:
        params["timeMin"] = time_min
    if isinstance(time_max, str) and time_max:
        params["timeMax"] = time_max

    raw_max = args.get("max_results")
    if raw_max is None:
        max_results = 50
    else:
        try:
            max_results = int(raw_max)
        except (TypeError, ValueError):
            raise ToolError(f"`max_results` must be an integer (got {raw_max!r}).")
    max_results = max(1, min(_MAX_RESULTS_CAP, max_results))
    params["maxResults"] = str(max_results)

    query = args.get("query")
    if isinstance(query, str) and query.strip():
        # Google's `q` is a full-text search over summary, description,
        # location, attendees. Useful for "find my standup".
        params["q"] = query.strip()

    resp = google_calendar_request(
        account, "GET", f"/calendars/{calendar_id}/events", params=params
    )
    if not resp.ok:
        raise calendar_api_error(resp, "list events")
    data = resp.json() or {}
    events = [shape_event(e) for e in data.get("items", []) if isinstance(e, dict)]
    return {
        "calendar_id": calendar_id,
        "from": time_min,
        "to": time_max,
        "events": events,
        "__summary__": f"Listed {len(events)} event(s) on `{calendar_id}`.",
    }


LIST_CALENDAR_EVENTS = Tool(
    name="list_calendar_events",
    description=(
        "List Google Calendar events in a date range. Use this when the "
        "user asks 'what's on my calendar', 'do I have anything Friday', "
        "'next meeting', 'find my standup'. Recurring events are expanded "
        "into single instances. Returns id, summary, start/end (ISO), "
        "attendees, hangout_link, html_link, status. Pass `query` for "
        "full-text search across summary/description/location/attendees. "
        "Defaults to the user's primary calendar."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "from": {
                "type": "STRING",
                "description": (
                    "Start of range as ISO 8601, e.g. "
                    "'2026-05-22T00:00:00Z'. Optional — omit for events "
                    "from the start of recorded time."
                ),
            },
            "to": {
                "type": "STRING",
                "description": (
                    "End of range as ISO 8601, e.g. "
                    "'2026-05-29T23:59:59Z'. Optional — omit for no upper "
                    "bound."
                ),
            },
            "query": {
                "type": "STRING",
                "description": (
                    "Optional full-text search over event summary, "
                    "description, location, and attendees."
                ),
            },
            "calendar_id": {
                "type": "STRING",
                "description": (
                    "Google calendar id. Omit for the user's primary "
                    "calendar; resolve via `list_calendars` for others."
                ),
            },
            "max_results": {
                "type": "INTEGER",
                "description": (
                    f"Cap the number of events returned. Default 50, " f"max {_MAX_RESULTS_CAP}."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
