"""Shared ACL helpers used by the agent tools.

These mirror — and consolidate — the membership-derivation logic that
the chunkers (`origin/search_engine/chunkers/*_chunker.py`) use to
populate each chunk's `acl_user_ids` field at index time. We re-derive
the same rules at fetch time so the agent can't pull entities the
requesting user isn't authorized to see, even if the LLM hallucinates
an id.

Each `chat_acl_user_ids` / `task_acl_user_ids` / `note_acl_user_ids`
returns the set of user UUIDs (as str) that should be allowed to read
the entity. The tools compare `ctx.user_id` to that set and raise
`ToolError` on mismatch.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError

from origin.models.chat.unified_models import Channel, ChannelMember
from origin.models.note.common_note_models import NotePermissionMaster
from origin.models.project.prj_models import ProjectMembers
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_PM,
    NOTE_TYPE_CHAT,
    NOTE_TYPE_PERSONAL,
    NOTE_TYPE_TASK,
)


def chat_acl_user_ids(chat_type_code: int, chat_id) -> set[str]:
    """Users allowed to read a v3 chat channel.

    `chat_type_code` is the integer kind (1=DM / 2=GM / 3=PM / 4=MDM),
    matching the values in `chunkers.base.CHAT_TYPE_*`. `chat_id` is the
    `Channel.id` UUID (str or UUID). Membership is resolved off the
    unified schema: DM/GM/MDM via `ChannelMember`; PM via
    `ProjectMembers` keyed on the channel's `project_id`.

    Returns an empty set for an unknown or malformed channel id (never
    raises) so callers can treat "no members" as "not authorized".
    """
    try:
        channel = Channel.objects.filter(id=chat_id, kind=chat_type_code, is_deleted=False).first()
    except (ValidationError, ValueError, TypeError):
        # Malformed UUID — treat as not found rather than raising.
        return set()
    if channel is None:
        return set()
    if chat_type_code == CHAT_TYPE_PM:
        return {
            str(uid)
            for uid in ProjectMembers.objects.filter(project_id=channel.project_id).values_list(
                "attendee_id", flat=True
            )
            if uid
        }
    return {
        str(uid)
        for uid in ChannelMember.objects.filter(channel=channel, is_deleted=False).values_list(
            "user_id", flat=True
        )
        if uid
    }


def task_acl_user_ids(project_id: int | None, assignee_id, reporter_id) -> set[str]:
    """Users allowed to read a task — project members + assignee + reporter."""
    out: set[str] = set()
    if project_id:
        out.update(
            str(uid)
            for uid in ProjectMembers.objects.filter(project_id=project_id).values_list(
                "attendee_id", flat=True
            )
            if uid
        )
    if assignee_id:
        out.add(str(assignee_id))
    if reporter_id:
        out.add(str(reporter_id))
    return out


def note_grants_user_ids(note_type_code: int, note_id: int) -> set[str]:
    """Users explicitly granted access to a note via `NotePermissionMaster`."""
    return {
        str(uid)
        for uid in NotePermissionMaster.objects.filter(
            note_type=note_type_code, note_id=note_id
        ).values_list("user_id", flat=True)
        if uid
    }


def chat_note_acl_user_ids(*, owner_id, chat_type_code: int, channel_id, note_id: int) -> set[str]:
    """Chat-note ACL: owner + channel members + explicit grants.

    Chat notes (`ChatNoteMaster`) are keyed on the v3 `Channel` UUID
    (`channel_id`), so membership resolves via the UUID-native
    `chat_acl_user_ids` (DM/GM/MDM via `ChannelMember`; PM via the
    channel's `project_id`).
    """
    out: set[str] = set()
    if owner_id:
        out.add(str(owner_id))
    out |= chat_acl_user_ids(chat_type_code, channel_id)
    out |= note_grants_user_ids(NOTE_TYPE_CHAT, note_id)
    return out


def task_note_acl_user_ids(*, owner_id, project_id: int | None, note_id: int) -> set[str]:
    """Task-note ACL: owner + project members + explicit grants."""
    out: set[str] = set()
    if owner_id:
        out.add(str(owner_id))
    if project_id:
        out.update(
            str(uid)
            for uid in ProjectMembers.objects.filter(project_id=project_id).values_list(
                "attendee_id", flat=True
            )
            if uid
        )
    out |= note_grants_user_ids(NOTE_TYPE_TASK, note_id)
    return out


def personal_note_acl_user_ids(*, owner_id, note_id: int) -> set[str]:
    """Personal-note ACL: owner + explicit grants only."""
    out: set[str] = set()
    if owner_id:
        out.add(str(owner_id))
    out |= note_grants_user_ids(NOTE_TYPE_PERSONAL, note_id)
    return out
