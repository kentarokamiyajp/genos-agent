"""Resolve viewer-friendly chat titles for search/source rows.

The chunker writes a viewer-agnostic placeholder into the chat `title`
field at index time (e.g. "DM 9", because a DM has no shared name — the
"name" is just the other participant, which depends on who's looking).
This module turns those placeholders into something a person can read:

  * DM  → the partner's username
  * GM  → the group's `group_name`
  * MDM → the group's `display_name`
  * PM  → the project's `project_name`

Used by:
  * `search.py` — after `_group_by_entity`, so typeahead responses
    show friendly names in result rows.
  * `agent/controller.py` — after each source-emitting tool call, so
    spotlight citation chips show friendly names.

Both call sites pass the requesting user's id (the viewer). DM
resolution is the only one that's viewer-dependent — the others
return the same name regardless — but threading user_id through the
single API keeps the call sites symmetric.
"""

from __future__ import annotations

from typing import Any


def friendly_chat_title(
    viewer_user_id: str,
    chat_type_label: Any,
    chat_id: Any,
) -> str | None:
    """Resolve a viewer-facing chat title; returns None on lookup failure.

    Best-effort: callers should treat None as "keep whatever title was
    already on the row" rather than blanking the field. A missing
    partner / soft-deleted chat / unknown chat-type label all return
    None silently.
    """
    if not chat_type_label or not chat_id:
        return None

    label = str(chat_type_label).lower()

    # Lazy imports — keep this module dependency-free at import time and
    # avoid circular-import surprises during Django startup. Chat identity
    # is the v3 `Channel.id` UUID the index/search rows carry; we resolve
    # the channel directly (the legacy integer bridge no longer applies).
    from django.core.exceptions import ValidationError

    from origin.models.chat.unified_models import Channel

    def _channel(kind: int) -> Channel | None:
        try:
            return Channel.objects.filter(id=chat_id, kind=kind, is_deleted=False).first()
        except (ValidationError, ValueError, TypeError):
            return None

    if label == "dm":
        from origin.models.chat.unified_models import ChannelMember
        from origin.models.common.user_models import CustomUser

        channel = _channel(1)  # CHAT_TYPE_DM
        if channel is None:
            return None
        member_ids = list(
            ChannelMember.objects.filter(channel=channel, is_deleted=False).values_list(
                "user_id", flat=True
            )
        )
        # The DM partner is the *other* member; self-DM falls back to self.
        partner_id = next((uid for uid in member_ids if str(uid) != viewer_user_id), None)
        if partner_id is None and member_ids:
            partner_id = member_ids[0]
        if not partner_id:
            return None
        try:
            return CustomUser.objects.get(id=partner_id).username or None
        except CustomUser.DoesNotExist:
            return None

    if label == "gm" or label == "mdm":
        # v3 `Channel.title` holds the group / MDM display name.
        channel = _channel(2 if label == "gm" else 4)
        return (channel.title or None) if channel else None

    if label == "pm":
        from origin.models.project.prj_models import ProjectMaster

        # chat_id is the PM channel's UUID; resolve it to its project.
        channel = _channel(3)  # CHAT_TYPE_PM
        if channel is None or not channel.project_id:
            return None
        try:
            return ProjectMaster.objects.get(project_id=channel.project_id).project_name or None
        except ProjectMaster.DoesNotExist:
            return None

    return None


def apply_friendly_titles(rows: list[dict[str, Any]], viewer_user_id: str) -> list[dict[str, Any]]:
    """Replace placeholder chat titles ('DM 9') with viewer-friendly names.

    Mutates and returns the same list. Only rows with `entity_type ==
    "chat"` are touched. Lookup failures leave the row's existing title
    in place.
    """
    for row in rows:
        if row.get("entity_type") != "chat":
            continue
        title = friendly_chat_title(viewer_user_id, row.get("chat_type"), row.get("chat_id"))
        if title:
            row["title"] = title
    return rows
