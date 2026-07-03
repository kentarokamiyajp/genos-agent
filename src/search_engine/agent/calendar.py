"""Shared helpers for the Google Calendar agent tools.

Five tools (`list_calendars`, `list_calendar_events`,
`create_calendar_event`, `update_calendar_event`, `delete_calendar_event`)
share the same auth check, request wrapper, and event-shaping logic.
Pulling it here keeps each tool file focused on its argument schema.

Design notes:
  - `resolve_google_calendar_account` raises `ToolError` directly so the
    model sees an actionable message. "Google not connected" and
    "calendar scope missing" map to different remediation buttons in
    the UI, and the LLM's natural-language answer should reflect that.
  - `shape_event` is the canonical projection of a Google event into a
    flat dict the model can scan. Free-form fields (summary, description)
    are wrapped via `wrap_workspace_content` because a malicious event
    body could try to inject instructions.
  - `build_time_endpoint` accepts the model's "YYYY-MM-DD" (all-day) or
    full ISO datetime and produces Google's start/end object shape.
    Asking the model to construct Google's exact schema was brittle —
    half the time it'd pass `dateTime` with a bare date string.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from origin.models.common.user_models import ConnectedAccount, CustomUser
from origin.search_engine.agent.tools.base import ToolError, wrap_workspace_content
from origin.services.oauth.tokens import ReauthRequired, get_valid_access_token

logger = logging.getLogger(__name__)

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
# Minimum scope for read AND write of calendar events. A user who only
# signed in with Google ("login" intent) has openid/email/profile and
# will hit upstream 403s on any Calendar v3 call — check up front so
# the model sees a clean signal instead of leaking the 403.
CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


def resolve_google_calendar_account(user_id: str) -> ConnectedAccount:
    """Resolve the user's Google ConnectedAccount, requiring calendar scope.

    Raises ToolError on:
      - user not found (defensive — shouldn't happen post-auth)
      - Google not connected for the user
      - account connected but missing the events scope
    """
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        raise ToolError("Current user record not found.")

    account = ConnectedAccount.objects.filter(user=user, provider="google").first()
    if account is None:
        raise ToolError(
            "Google is not connected for this user. Direct them to "
            "Settings → Integrations to connect Google."
        )
    if CALENDAR_EVENTS_SCOPE not in (account.scopes or []):
        raise ToolError(
            "Google is connected but Calendar access hasn't been granted. "
            "Direct the user to Settings → Tasks → Grant Calendar access "
            "(or the matching button in Integrations)."
        )
    return account


def google_calendar_request(
    account: ConnectedAccount,
    method: str,
    path: str,
    **kwargs: Any,
) -> requests.Response:
    """`requests` wrapper that injects the auto-refreshed access token."""
    try:
        token = get_valid_access_token(account)
    except ReauthRequired as exc:
        # Refresh token is revoked/expired — same actionable shape as the
        # not-connected / scope-missing ToolErrors in
        # `resolve_google_calendar_account`, so raise the same kind of
        # message rather than leaking a stack trace into the agent.
        raise ToolError(
            "Google Calendar access has expired. Direct the user to "
            "Settings → Integrations to reconnect Google."
        ) from exc
    headers = kwargs.pop("headers", {}) or {}
    headers["Authorization"] = f"Bearer {token}"
    headers.setdefault("Accept", "application/json")
    return requests.request(
        method, f"{CALENDAR_API_BASE}{path}", headers=headers, timeout=15, **kwargs
    )


def _is_date_only(value: str) -> bool:
    """True for "YYYY-MM-DD". False for ISO datetimes."""
    return isinstance(value, str) and len(value) == 10 and "T" not in value


def build_time_endpoint(value: str, time_zone: str | None) -> dict[str, Any]:
    """Turn a user-supplied ISO string into Google's start/end shape.

    - "YYYY-MM-DD" → all-day: {"date": "YYYY-MM-DD"}
    - "YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]" → timed: {"dateTime": ..., "timeZone": ...}
    """
    if _is_date_only(value):
        return {"date": value}
    body: dict[str, Any] = {"dateTime": value}
    if time_zone:
        body["timeZone"] = time_zone
    return body


def validate_time_pair(start_iso: str, end_iso: str) -> None:
    """Reject mixed all-day/timed endpoint shapes; Google rejects them too."""
    if _is_date_only(start_iso) != _is_date_only(end_iso):
        raise ToolError(
            "`start_iso` and `end_iso` must both be date-only (YYYY-MM-DD) "
            "or both be ISO datetimes — Google rejects mixed shapes."
        )


def emails_to_attendees(raw: Any) -> list[dict[str, str]]:
    """Convert an LLM-supplied array of email strings into Google's
    attendees shape. Non-string entries and empty strings are dropped.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            out.append({"email": entry.strip()})
    return out


def shape_event(event: dict[str, Any]) -> dict[str, Any]:
    """Project a Google Calendar event into a flat, LLM-friendly dict."""
    start = event.get("start") or {}
    end = event.get("end") or {}
    attendees_raw = event.get("attendees") or []
    attendees = [
        {
            "email": a.get("email"),
            "displayName": a.get("displayName"),
            "responseStatus": a.get("responseStatus"),
        }
        for a in attendees_raw
        if isinstance(a, dict) and a.get("email")
    ]
    return {
        "id": event.get("id"),
        "summary": wrap_workspace_content(event.get("summary") or ""),
        "description": wrap_workspace_content(event.get("description") or ""),
        "location": event.get("location"),
        "start": start.get("date") or start.get("dateTime"),
        "end": end.get("date") or end.get("dateTime"),
        "all_day": "date" in start,
        "html_link": event.get("htmlLink"),
        "hangout_link": event.get("hangoutLink"),
        "attendees": attendees,
        "status": event.get("status"),
        "creator_email": (event.get("creator") or {}).get("email"),
        "organizer_email": (event.get("organizer") or {}).get("email"),
    }


def calendar_api_error(resp: requests.Response, context: str) -> ToolError:
    """Build a uniform ToolError from a non-OK calendar response.

    `context` is a short verb phrase like "create event" / "list events"
    that gets folded into the error message the model sees.
    """
    logger.warning(
        "calendar tool %s failed status=%s body=%s",
        context,
        resp.status_code,
        resp.text[:500],
    )
    return ToolError(f"Google Calendar {context} failed: HTTP {resp.status_code}.")
