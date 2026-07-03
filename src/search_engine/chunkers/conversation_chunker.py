"""Chunker for completed `AgentRun` rows — the vector conversation-memory
lane (C1 / Q2.3, SPOTLIGHT_QUALITY_ARCHITECTURE.md §4.7).

One chunk per clean, grounded agent answer. This is a SEPARATE, per-user
lane: `entity_type="conversation"` and `acl_user_ids=[run.user_id]`, so a
user only ever recalls their OWN past conversations, and normal workspace
search (`search()` with no entity_types) excludes the lane entirely — it is
reachable only via the `search_past_conversations` tool, which passes
`entity_types=["conversation"]` explicitly.

Like the note/thread summary chunkers, conversations are indexed by the
batch reindex (`ingest_all(entity_types=["conversation"])` /
`opensearch_reindex`), NOT on the answer path — so enabling memory adds zero
latency to the live ask.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from origin.search_engine.chunkers.base import (
    Chunk,
    EntityChunks,
    iso,
    make_snippet,
)
from origin.search_engine.models import AgentRun

# Only the clean-completion signal is worth remembering. Mirrors the F2
# sampler's scope (`agent_judge_sample`): rejected / step_cap / error runs
# are not durable memory.
_INDEXABLE_STATUS = "done"
# Bound the search_text so one runaway answer doesn't dominate the embedding.
_MAX_ANSWER_CHARS = 4000


def iter_conversation_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    qs = AgentRun.objects.filter(status=_INDEXABLE_STATUS).exclude(final_answer_text="")
    if since is not None:
        # finished_at is the completion time; fall back to started_at for
        # rows finished before that column was populated.
        qs = qs.filter(started_at__gte=since)
    qs = qs.order_by("started_at")

    for run in qs.iterator():
        query = (run.query or "").strip()
        answer = (run.final_answer_text or "").strip()
        user_id = str(run.user_id or "")
        team_id = str(run.team_id or "")
        if not query or not answer or not user_id or not team_id:
            continue

        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + "…"

        entity_id = f"conversation:{run.run_id}"
        # search_text carries BOTH the question and the answer so a recall
        # query ("what did I decide about the perf budget?") matches on
        # either side. The title is the original question — the most useful
        # one-line handle when the chunk surfaces as a source.
        search_text = f"Q: {query}\nA: {answer}"

        chunk = Chunk(
            chunk_id=entity_id,
            entity_type="conversation",
            entity_id=entity_id,
            chunk_type="conversation",
            team_id=team_id,
            # Per-user lane: only the asker can recall their own past Q&A.
            acl_user_ids=[user_id],
            title=query if len(query) <= 200 else query[:197] + "…",
            search_text=search_text,
            snippet_text=make_snippet(answer),
            related_entity_ids=[],
            created_at=iso(run.started_at),
            updated_at=iso(run.finished_at or run.started_at),
        )
        yield EntityChunks(
            entity_type="conversation",
            entity_id=entity_id,
            chunks=[chunk],
        )
