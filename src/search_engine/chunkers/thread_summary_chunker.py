"""Chunker for `ThreadSummary` rows.

One chunk per thread summary. ACL is derived from the underlying chat's
members (same logic the agent's `chat_acl_user_ids` uses, so the index
view matches the live fetch view).

Entity_type is `"thread_summary"` so the wider Spotlight search can
distinguish thread summaries from chat messages and notes when ranking
results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from origin.search_engine.agent.acl import chat_acl_user_ids
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_LABEL,
    Chunk,
    EntityChunks,
    chat_entity_id,
    iso,
    make_snippet,
)
from origin.search_engine.models import ThreadSummary


def iter_thread_summary_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = ThreadSummary.objects.all().order_by("id")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)

    for summary in qs.iterator():
        if not summary.team_id or not summary.summary_text:
            continue

        chat_label = CHAT_TYPE_LABEL.get(summary.chat_type)
        if not chat_label:
            continue

        acl_user_ids = sorted(chat_acl_user_ids(summary.chat_type, str(summary.chat_id)))
        # An empty ACL would render the chunk unsearchable for everyone.
        # That's actually correct (no members → no readers), but emitting
        # an empty acl_user_ids list lets the indexer's ACL filter accept
        # *any* user — defensively skip the chunk instead.
        if not acl_user_ids:
            continue

        entity_id = f"thread_summary:{summary.chat_type}:{summary.chat_id}:{summary.thread_id}"
        chunk = Chunk(
            chunk_id=entity_id,
            entity_type="thread_summary",
            entity_id=entity_id,
            chunk_type="thread_summary",
            team_id=str(summary.team_id),
            acl_user_ids=acl_user_ids,
            title=f"Thread summary ({chat_label}:{summary.chat_id} thread {summary.thread_id})",
            search_text=summary.summary_text,
            snippet_text=make_snippet(summary.summary_text),
            related_entity_ids=[
                chat_entity_id(chat_label, summary.chat_id, summary.thread_id),
            ],
            chat_type=chat_label,
            chat_id=str(summary.chat_id),
            thread_id=str(summary.thread_id),
            created_at=iso(summary.ts_created_at),
            updated_at=iso(summary.ts_updated_at),
        )
        yield EntityChunks(
            entity_type="thread_summary",
            entity_id=entity_id,
            chunks=[chunk],
        )
