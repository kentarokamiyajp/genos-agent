"""`get_my_schedule` tool — composite "what's on my plate" view.

Combines two surfaces in one call:
  * `tasks_due` — tasks assigned to me with a `due_date` in the window.
  * `calendar_events` — Google Calendar events in the same window from
    the user's primary calendar.

Answers "task schedule for this week", "what's on my plate tomorrow",
"my next 14 days". The agent gets a single ranked picture without
having to fan out to `list_tasks` + `list_calendar_events` separately.

Date window:
  * If both `from` and `to` are omitted, defaults to today 00:00 →
    today+7d 23:59 in the server's timezone.
  * If only one is given, the other is bounded to ±7 days from the
    given side so the window is always finite.

Calendar half is best-effort: if the user hasn't connected Google or
granted Calendar scope, `calendar_events` returns `[]` with
`calendar_status="not_connected"` rather than failing the whole call.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.calendar import (
    calendar_api_error,
    google_calendar_request,
    resolve_google_calendar_account,
    shape_event,
)
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_DEFAULT_WINDOW_DAYS = 7
_MAX_TASKS = 50
_MAX_EVENTS = 50


def _parse_iso(value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"`{field_name}` must be an ISO 8601 string (got {value!r}).")
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        raise ToolError(
            f"`{field_name}` must be ISO 8601 (e.g. '2026-05-27T00:00:00Z'); got {value!r}."
        )
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed)
    return parsed


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_from = args.get("from")
    raw_to = args.get("to")
    from_dt = _parse_iso(raw_from, "from")
    to_dt = _parse_iso(raw_to, "to")

    now = timezone.now()
    today = now.date()
    if from_dt is None and to_dt is None:
        from_dt = timezone.make_aware(datetime.combine(today, time.min))
        to_dt = from_dt + timedelta(days=_DEFAULT_WINDOW_DAYS)
    elif from_dt is None:
        from_dt = to_dt - timedelta(days=_DEFAULT_WINDOW_DAYS)
    elif to_dt is None:
        to_dt = from_dt + timedelta(days=_DEFAULT_WINDOW_DAYS)

    if from_dt > to_dt:
        raise ToolError("`from` must be before `to`.")

    from_date = from_dt.date()
    to_date = to_dt.date()

    # --- Tasks due in window, assigned to me. ---
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    tasks_qs = (
        TaskMaster.objects.filter(
            team_id=ctx.team_id,
            assignee_id=ctx.user_id,
            project_id__in=member_project_ids,
            is_deleted=False,
            is_init_task=False,
            due_date__isnull=False,
            due_date__gte=from_date,
            due_date__lte=to_date,
        )
        .exclude(status__in=["Closed", "Deleted"])
        .select_related("project")
        .order_by("due_date", "-ts_updated_at")[:_MAX_TASKS]
    )

    tasks_due: list[dict[str, Any]] = []
    for t in tasks_qs:
        tasks_due.append(
            {
                "task_id": t.task_id,
                "display_id": t.display_id,
                "title": t.title,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "status": t.status,
                "priority": t.priority,
                "project_id": t.project_id,
                "project_name": t.project.project_name if t.project else None,
            }
        )

    # --- Calendar events in window. Best-effort — graceful "no calendar"
    #     fallback so the task half still ships. ---
    calendar_events: list[dict[str, Any]] = []
    calendar_status = "ok"
    calendar_message: str | None = None

    try:
        account = resolve_google_calendar_account(ctx.user_id)
    except ToolError as exc:
        calendar_status = "not_connected"
        calendar_message = str(exc)
        account = None

    if account is not None:
        params: dict[str, Any] = {
            "singleEvents": "true",
            "orderBy": "startTime",
            "timeMin": from_dt.isoformat(),
            "timeMax": to_dt.isoformat(),
            "maxResults": str(_MAX_EVENTS),
        }
        resp = google_calendar_request(account, "GET", "/calendars/primary/events", params=params)
        if not resp.ok:
            # Surface as a soft warning, not a hard error — the task half
            # is still valuable on its own.
            calendar_status = "error"
            calendar_message = f"Calendar fetch failed: {resp.status_code}"
            try:
                raise calendar_api_error(resp, "list events")
            except ToolError as exc:
                calendar_message = str(exc)
        else:
            data = resp.json() or {}
            calendar_events = [
                shape_event(e) for e in data.get("items", []) if isinstance(e, dict)
            ]

    summary_bits = [
        f"{len(tasks_due)} task(s) due",
        f"{len(calendar_events)} calendar event(s)",
    ]
    if calendar_status != "ok":
        summary_bits.append(f"calendar: {calendar_status}")
    summary = f"{from_date.isoformat()} → {to_date.isoformat()}: " + ", ".join(summary_bits)

    return {
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "tasks_due": tasks_due,
        "calendar_events": calendar_events,
        "calendar_status": calendar_status,
        "calendar_message": calendar_message,
        "__summary__": summary,
    }


GET_MY_SCHEDULE = Tool(
    name="get_my_schedule",
    description=(
        "Composite 'my schedule' view: tasks assigned to me with a "
        "due_date in the window, plus Google Calendar events in the "
        "same window. Use for 'what's my schedule this week?', 'task "
        "schedule for the next few days', 'what's on my plate "
        "tomorrow?'. Defaults to today → today+7 days. If the user "
        "hasn't connected Google Calendar, calendar_events is `[]` and "
        "calendar_status is 'not_connected' — tasks are still returned. "
        "Pair with `get_my_focus_tasks` if the user wants priority-"
        "ranked next-ups (this tool is date-ranked)."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "from": {
                "type": "STRING",
                "description": (
                    "Window start as ISO 8601 datetime, e.g. "
                    "'2026-05-27T00:00:00Z'. Omit to default to today "
                    "00:00 in the server's timezone."
                ),
            },
            "to": {
                "type": "STRING",
                "description": (
                    "Window end as ISO 8601 datetime, e.g. "
                    "'2026-06-03T23:59:59Z'. Omit to default to "
                    "from + 7 days."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
