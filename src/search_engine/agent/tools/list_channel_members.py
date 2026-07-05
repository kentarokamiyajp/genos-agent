"""`list_channel_members` tool — who is in ONE chat channel (A1-ext).

Answers "who's in this group chat / project chat?" from the unified
`ChannelMember` table (which subsumed the legacy GM/MDM/DM membership
tables in migration 0132). Note there is NO thread-membership table —
a thread is a message subtree, and "who's in the thread" is either the
parent channel's roster (this tool) or the distinct senders inside it
(readable via fetch_chat_thread).

ACL contract:
  * Tenant guard: channel.team_id must equal ctx.team_id.
  * Membership guard: the requester must hold a non-deleted
    ChannelMember row on the channel — non-members may not enumerate a
    private channel's roster.
  * Soft-deleted memberships and system/deleted users are excluded.
"""

from __future__ import annotations

import uuid as uuid_mod
from typing import Any

from origin.models.chat.unified_models import Channel, ChannelMember
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_KIND_LABELS = {1: "dm", 2: "gm", 3: "pm", 4: "mdm"}


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_channel_id = args.get("channel_id")
    try:
        channel_id = uuid_mod.UUID(str(raw_channel_id))
    except (TypeError, ValueError):
        raise ToolError(f"`channel_id` must be a UUID (got {raw_channel_id!r}).")

    try:
        channel = Channel.objects.get(id=channel_id, is_deleted=False)
    except Channel.DoesNotExist:
        raise ToolError(f"Channel {channel_id} not found.")

    if str(getattr(channel, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: channel belongs to a different team.")

    memberships = ChannelMember.objects.filter(
        channel=channel, is_deleted=False
    ).select_related("user")
    if not any(str(m.user_id) == ctx.user_id for m in memberships):
        raise ToolError(
            f"Not authorized to access channel {channel_id}. "
            "You are not a member of that channel."
        )

    members = []
    for m in memberships:
        u = m.user
        if u is None or u.is_deleted or u.is_system_user:
            continue
        members.append(
            {
                "user_id": str(u.id),
                "username": u.username or "",
                "role": m.role or "member",
            }
        )

    title = channel.title or _KIND_LABELS.get(channel.kind, "chat")
    return {
        "channel_id": str(channel.id),
        "channel_kind": _KIND_LABELS.get(channel.kind, str(channel.kind)),
        "channel_title": channel.title or "",
        "members": members,
        "__summary__": f"{len(members)} member(s) in {title}",
    }


LIST_CHANNEL_MEMBERS = Tool(
    name="list_channel_members",
    description=(
        "List the members of ONE chat channel (user_id, username, role) "
        "by its channel_id UUID. Use this for 'who is in this group "
        "chat / project chat?' questions. You must be a member of the "
        "channel yourself. For 'who is on the whole team' use "
        "get_team_members; for project rosters use list_project_members."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "channel_id": {
                "type": "STRING",
                "description": "The channel's UUID (e.g. from a chat citation or fetch_chat_thread).",
            },
        },
        "required": ["channel_id"],
    },
    run=_run,
)
