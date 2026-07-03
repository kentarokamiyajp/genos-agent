"""Chat chunker covering DM, GM, MDM, PM — unified Channel/Message schema.

Sources from the v3 unified messaging tables (`Channel` + `Message`),
which replaced the per-type `DMMessages` / `GMMessages` / `MDMMessages` /
`PMMessages` tables (dropped in migration `0132_drop_legacy_chat_models`).
Chat identity is the UUID pair `Channel.id` (chat_id) / `Message.id`
(message id + thread-root id), NOT the legacy integer ids.

For each chat *thread* (including the main non-thread channel as
thread_id=None), we produce:

  - One `chat_message` chunk per individual message (good for keyword
    search and short queries).
  - One `chat_thread_window` chunk concatenating every message in the
    thread, in order (good for semantic / natural-language search and
    to give RAG enough context).

A top-level message that roots a thread (has replies) is represented
inside its thread (as the `anchor` chunk), not in the main channel —
the two sets are mutually exclusive so a message's text is indexed once.

Preceding-message context: each `chat_message` chunk's `search_text` is
prefixed with the previous N messages from the same channel/thread
(default N=2, via `RAG_CHAT_CONTEXT_WINDOW`). This gives the embedding
lane real conversational context so terse replies like "yes, ship it"
embed near related preceding messages. The `snippet_text` stays focused
on the focal message so the UI doesn't show prior text as the matched
content.

ACL is denormalized per chunk: the chat's allowed user list is copied
into `acl_user_ids` so retrieval-time filtering is a single `terms`
clause. DM/GM/MDM membership comes from `ChannelMember`; PM membership
comes from `ProjectMembers` keyed on the channel's `project_id`.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator, Optional

from django.conf import settings

from origin.models.chat.unified_models import Channel, ChannelMember, Message
from origin.models.common.user_models import CustomUser
from origin.models.project.prj_models import ProjectMembers
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_DM,
    CHAT_TYPE_GM,
    CHAT_TYPE_LABEL,
    CHAT_TYPE_MDM,
    CHAT_TYPE_PM,
    Chunk,
    EntityChunks,
    chat_entity_id,
    iso,
    make_snippet,
)
from origin.search_engine.models import ThreadSummary
from origin.search_engine.text_extraction import extract_text


def iter_dm_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from _iter_kind_chunks(CHAT_TYPE_DM, "dm", since)


def iter_gm_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from _iter_kind_chunks(CHAT_TYPE_GM, "gm", since)


def iter_mdm_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from _iter_kind_chunks(CHAT_TYPE_MDM, "mdm", since)


def iter_pm_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from _iter_kind_chunks(CHAT_TYPE_PM, "pm", since)


def iter_all_chat_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    yield from iter_dm_chunks(since)
    yield from iter_gm_chunks(since)
    yield from iter_mdm_chunks(since)
    yield from iter_pm_chunks(since)


# --------------------------------------------------------------------------- #
# Per-kind driver                                                             #
# --------------------------------------------------------------------------- #


def _iter_kind_chunks(
    kind_code: int, label: str, since: Optional[datetime]
) -> Iterator[EntityChunks]:
    """Yield one EntityChunks per (channel, thread) for a single chat kind.

    With `since`, only channels with a message updated at/after `since`
    are re-emitted (the whole channel re-chunks; ingestion dedupes the
    unchanged chunks by hash)."""
    channels = Channel.objects.filter(kind=kind_code, is_deleted=False).select_related(
        "project", "team"
    )

    if since is not None:
        dirty_channel_ids = set(
            Message.objects.filter(channel__kind=kind_code, ts_updated_at__gte=since).values_list(
                "channel_id", flat=True
            )
        )
        channels = channels.filter(id__in=dirty_channel_ids)

    channel_list = list(channels)
    if not channel_list:
        return
    channel_ids = [c.id for c in channel_list]

    # Messages per channel (non-deleted), ordered by the monotonic seq so
    # the context window and thread window read chronologically.
    msgs_by_channel: dict[object, list[Message]] = defaultdict(list)
    for m in (
        Message.objects.filter(channel_id__in=channel_ids, deleted_at__isnull=True)
        .select_related("sender", "task")
        .order_by("channel_id", "seq")
    ):
        msgs_by_channel[m.channel_id].append(m)

    # ACL preload.
    is_pm = kind_code == CHAT_TYPE_PM
    members_by_channel: dict[object, list[str]] = defaultdict(list)
    members_by_project: dict[int, list[str]] = defaultdict(list)
    if is_pm:
        project_ids = [c.project_id for c in channel_list if c.project_id]
        for row in ProjectMembers.objects.filter(project_id__in=project_ids).values(
            "project_id", "attendee_id"
        ):
            if row["attendee_id"] is not None:
                members_by_project[row["project_id"]].append(str(row["attendee_id"]))
    else:
        for row in ChannelMember.objects.filter(
            channel_id__in=channel_ids, is_deleted=False
        ).values("channel_id", "user_id"):
            if row["user_id"] is not None:
                members_by_channel[row["channel_id"]].append(str(row["user_id"]))

    # Pre-resolve sender names + summarized threads in one pass.
    sender_ids: set = set()
    for msgs in msgs_by_channel.values():
        for m in msgs:
            if m.sender_id:
                sender_ids.add(m.sender_id)
    sender_names = _load_sender_names(sender_ids)
    summarized = _load_summarized_threads()

    for channel in channel_list:
        if not channel.team_id:
            continue
        msgs = msgs_by_channel.get(channel.id, [])
        if not msgs:
            continue
        if is_pm:
            acl_user_ids = members_by_project.get(channel.project_id, [])
            project_id = str(channel.project_id) if channel.project_id else None
        else:
            acl_user_ids = members_by_channel.get(channel.id, [])
            project_id = None

        # An empty ACL would index chunks no user can retrieve (and, if the
        # retrieval-time ACL filter ever treats an empty `terms` list as
        # match-any, could leak the chunk to everyone). Skip the channel —
        # same defensive stance as thread_summary_chunker / note_summary_chunker.
        if not acl_user_ids:
            continue

        yield from _emit_channel_chunks(
            channel=channel,
            label=label,
            team_id=str(channel.team_id),
            acl_user_ids=acl_user_ids,
            messages=msgs,
            project_id=project_id,
            sender_names=sender_names,
            summarized=summarized,
        )


# --------------------------------------------------------------------------- #
# Emission                                                                    #
# --------------------------------------------------------------------------- #


def _emit_channel_chunks(
    *,
    channel: Channel,
    label: str,
    team_id: str,
    acl_user_ids: list[str],
    messages: list[Message],
    project_id: Optional[str],
    sender_names: dict[str, str],
    summarized: set[tuple[str, str, str]],
) -> Iterator[EntityChunks]:
    """One EntityChunks per (channel main timeline) + per thread."""
    channel_uuid = str(channel.id)
    chat_title = channel.title or _placeholder_title(label, channel)

    top_level = [m for m in messages if not m.is_thread_reply]
    replies_by_root: dict[object, list[Message]] = defaultdict(list)
    for m in messages:
        if m.is_thread_reply and m.thread_root_id:
            replies_by_root[m.thread_root_id].append(m)
    root_ids = set(replies_by_root.keys())
    top_by_id = {m.id: m for m in top_level}

    # 1) Main channel: top-level messages that do NOT root a thread.
    plain_main = [m for m in top_level if m.id not in root_ids]
    main_entity_id = chat_entity_id(label, channel_uuid)
    main_chunks = _build_message_chunks(
        label=label,
        channel_uuid=channel_uuid,
        thread_id=None,
        team_id=team_id,
        acl_user_ids=acl_user_ids,
        chat_title=chat_title,
        entity_id=main_entity_id,
        messages=plain_main,
        project_id=project_id,
        sender_names=sender_names,
    )
    if main_chunks:
        yield EntityChunks(entity_type="chat", entity_id=main_entity_id, chunks=main_chunks)

    # 2) One entity per thread (rooted at a top-level message).
    for root_id in root_ids:
        anchor = top_by_id.get(root_id)  # None if the root was soft-deleted
        replies = replies_by_root[root_id]
        root_uuid = str(root_id)
        thread_entity_id = chat_entity_id(label, channel_uuid, root_uuid)
        skip_window = (label, channel_uuid, root_uuid) in summarized
        chunks = _build_thread_chunks(
            label=label,
            channel_uuid=channel_uuid,
            root_uuid=root_uuid,
            team_id=team_id,
            acl_user_ids=acl_user_ids,
            chat_title=chat_title,
            entity_id=thread_entity_id,
            anchor=anchor,
            replies=replies,
            project_id=project_id,
            sender_names=sender_names,
            skip_window=skip_window,
        )
        if chunks:
            yield EntityChunks(entity_type="chat", entity_id=thread_entity_id, chunks=chunks)


def _build_message_chunks(
    *,
    label: str,
    channel_uuid: str,
    thread_id: Optional[str],
    team_id: str,
    acl_user_ids: list[str],
    chat_title: str,
    entity_id: str,
    messages: list[Message],
    project_id: Optional[str],
    sender_names: dict[str, str],
) -> list[Chunk]:
    out: list[Chunk] = []
    context_size = _context_window_size()
    recent_texts: list[str] = []
    for m in messages:
        text = extract_text(m.body)
        if not text:
            continue
        related = [f"task:{m.task_id}"] if m.task_id else []
        msg_uuid = str(m.id)
        prior = recent_texts[-context_size:] if context_size else []
        sid = str(m.sender_id) if m.sender_id else None
        out.append(
            Chunk(
                chunk_id=f"chat:{label}:{channel_uuid}:msg:{msg_uuid}",
                entity_type="chat",
                entity_id=entity_id,
                chunk_type="chat_message",
                team_id=team_id,
                acl_user_ids=acl_user_ids,
                title=chat_title,
                search_text=_search_text_with_context(text, prior),
                snippet_text=make_snippet(text),
                related_entity_ids=related,
                chat_type=label,
                chat_id=channel_uuid,
                thread_id=thread_id,
                project_id=project_id,
                author_id=sid,
                author_name=sender_names.get(sid) if sid else None,
                chat_message_id=msg_uuid,
                created_at=iso(m.ts_sent_at),
                updated_at=iso(m.ts_updated_at),
            )
        )
        recent_texts.append(text)
    return out


def _build_thread_chunks(
    *,
    label: str,
    channel_uuid: str,
    root_uuid: str,
    team_id: str,
    acl_user_ids: list[str],
    chat_title: str,
    entity_id: str,
    anchor: Optional[Message],
    replies: list[Message],
    project_id: Optional[str],
    sender_names: dict[str, str],
    skip_window: bool,
) -> list[Chunk]:
    out: list[Chunk] = []
    window_parts: list[str] = []
    related: set[str] = set()
    latest_ts = None
    context_size = _context_window_size()
    recent_texts: list[str] = []

    if anchor is not None:
        text = extract_text(anchor.body)
        if text:
            window_parts.append(text)
            sid = str(anchor.sender_id) if anchor.sender_id else None
            # The anchor's chat_message_id == the thread root id.
            out.append(
                Chunk(
                    chunk_id=f"chat:{label}:{channel_uuid}:thread:{root_uuid}:anchor:{root_uuid}",
                    entity_type="chat",
                    entity_id=entity_id,
                    chunk_type="chat_message",
                    team_id=team_id,
                    acl_user_ids=acl_user_ids,
                    title=chat_title,
                    search_text=text,
                    snippet_text=make_snippet(text),
                    related_entity_ids=[f"task:{anchor.task_id}"] if anchor.task_id else [],
                    chat_type=label,
                    chat_id=channel_uuid,
                    thread_id=root_uuid,
                    project_id=project_id,
                    author_id=sid,
                    author_name=sender_names.get(sid) if sid else None,
                    chat_message_id=root_uuid,
                    created_at=iso(anchor.ts_sent_at),
                    updated_at=iso(anchor.ts_updated_at),
                )
            )
            if anchor.task_id:
                related.add(f"task:{anchor.task_id}")
            latest_ts = anchor.ts_updated_at
            recent_texts.append(text)

    for r in replies:
        text = extract_text(r.body)
        if not text:
            continue
        window_parts.append(text)
        prior = recent_texts[-context_size:] if context_size else []
        sid = str(r.sender_id) if r.sender_id else None
        reply_uuid = str(r.id)
        out.append(
            Chunk(
                chunk_id=f"chat:{label}:{channel_uuid}:thread:{root_uuid}:msg:{reply_uuid}",
                entity_type="chat",
                entity_id=entity_id,
                chunk_type="chat_message",
                team_id=team_id,
                acl_user_ids=acl_user_ids,
                title=chat_title,
                search_text=_search_text_with_context(text, prior),
                snippet_text=make_snippet(text),
                related_entity_ids=[],
                chat_type=label,
                chat_id=channel_uuid,
                thread_id=root_uuid,
                project_id=project_id,
                author_id=sid,
                author_name=sender_names.get(sid) if sid else None,
                chat_message_id=reply_uuid,
                created_at=iso(r.ts_sent_at),
                updated_at=iso(r.ts_updated_at),
            )
        )
        recent_texts.append(text)
        if r.ts_updated_at and (latest_ts is None or r.ts_updated_at > latest_ts):
            latest_ts = r.ts_updated_at

    # Thread-window chunk: concatenated text for semantic search.
    # Suppressed for threads that already have a `ThreadSummary` row; the
    # LLM abstract supersedes raw concatenation for vector recall.
    if window_parts and not skip_window:
        window_text = "\n".join(window_parts)
        out.append(
            Chunk(
                chunk_id=f"chat:{label}:{channel_uuid}:thread:{root_uuid}:window",
                entity_type="chat",
                entity_id=entity_id,
                chunk_type="chat_thread_window",
                team_id=team_id,
                acl_user_ids=acl_user_ids,
                title=chat_title,
                search_text=window_text,
                snippet_text=make_snippet(window_text),
                related_entity_ids=sorted(related),
                chat_type=label,
                chat_id=channel_uuid,
                thread_id=root_uuid,
                project_id=project_id,
                created_at=iso(anchor.ts_sent_at) if anchor else None,
                updated_at=iso(latest_ts),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _placeholder_title(label: str, channel: Channel) -> str:
    """Viewer-agnostic fallback title; `friendly_titles` resolves the real
    DM-partner / GM / MDM / PM name at query time."""
    if label == "pm":
        project = getattr(channel, "project", None)
        if project is not None and project.project_name:
            return project.project_name
        return "Project chat"
    if label == "dm":
        return "Direct message"
    return "Group chat"


def _context_window_size() -> int:
    """How many preceding messages to fold into each chat_message chunk."""
    try:
        return max(0, int(settings.SEARCH_ENGINE.get("RAG_CHAT_CONTEXT_WINDOW", 2)))
    except (ValueError, TypeError):
        return 2


def _search_text_with_context(focal_text: str, prior_texts: list[str]) -> str:
    """Build the `search_text` for a chat_message chunk.

    With context disabled (`RAG_CHAT_CONTEXT_WINDOW=0`) or no prior
    messages (first message in a channel/thread), returns the focal text
    unchanged so the hash-diff in `ingestion.py` stays stable.
    """
    if not prior_texts:
        return focal_text
    prior = "\n".join(prior_texts)
    return f"Previously:\n{prior}\n\nMessage:\n{focal_text}"


def _load_sender_names(sender_ids: set) -> dict[str, str]:
    """Batch-resolve sender_id → display name for the `author_name`
    denormalization. Missing/deleted users fall back to ""."""
    out: dict[str, str] = {}
    clean = [s for s in sender_ids if s]
    if not clean:
        return out
    for u in CustomUser.objects.filter(id__in=clean).values("id", "username"):
        out[str(u["id"])] = u["username"] or ""
    return out


def _load_summarized_threads() -> set[tuple[str, str, str]]:
    """Return {(chat_label, chat_id, thread_id)} (UUID strings) that already
    have a `ThreadSummary` row, so the chunker can skip emitting a
    `chat_thread_window` chunk for them — the indexed abstract
    (entity_type="thread_summary") is strictly better for vector recall."""
    out: set[tuple[str, str, str]] = set()
    for row in ThreadSummary.objects.all().values("chat_type", "chat_id", "thread_id"):
        label = CHAT_TYPE_LABEL.get(row["chat_type"])
        if not label:
            continue
        out.add((label, str(row["chat_id"]), str(row["thread_id"])))
    return out
