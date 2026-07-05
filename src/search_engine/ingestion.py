"""Ingestion orchestrator.

Walks `EntityChunks` produced by each chunker, and for every entity:

  1. Hashes the chunk text.
  2. Compares against the `RagChunk` tracking table to split chunks
     into `new` / `changed` / `unchanged` and identifies stale chunk
     ids that exist in the table but no longer in the regeneration.
  3. Embeds only new + changed chunks (skipping unchanged saves
     OpenAI cost and time).
  4. Bulk-indexes the new + changed docs into OpenSearch.
  5. Deletes stale chunks from OpenSearch.
  6. Mirrors all of that into `RagChunk` (upsert / delete) so the
     next run can repeat steps 1-2.

This is the entry point both the `opensearch_reindex` management
command and (later) any Kafka-style indexer worker should call.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Optional

from django.conf import settings
from django.db import transaction
from opensearchpy import helpers as os_helpers

from origin.search_engine.chunkers.base import Chunk, EntityChunks
from origin.search_engine.chunkers.chat_chunker import iter_all_chat_chunks
from origin.search_engine.chunkers.conversation_chunker import (
    conversation_chunks_for_run,
    iter_conversation_chunks,
)
from origin.search_engine.chunkers.milestone_chunker import iter_milestone_chunks
from origin.search_engine.chunkers.note_chunker import iter_all_note_chunks
from origin.search_engine.chunkers.note_summary_chunker import iter_note_summary_chunks
from origin.search_engine.chunkers.spotlight_answer_chunker import iter_spotlight_answer_chunks
from origin.search_engine.chunkers.task_chunker import iter_task_chunks
from origin.search_engine.chunkers.thread_summary_chunker import iter_thread_summary_chunks
from origin.search_engine.chunkers.todo_chunker import iter_todo_chunks
from origin.search_engine.embeddings import (
    embed_texts,
    get_active_embedding_model_name,
    hash_text,
)
from origin.search_engine.index_config import INDEX_SCHEMA_VERSION
from origin.search_engine.models import RagChunk
from origin.search_engine.opensearch_client import get_client, get_index_alias

log = logging.getLogger(__name__)


class IngestionStats:
    def __init__(self):
        self.entities_processed = 0
        self.chunks_total = 0
        self.chunks_new = 0
        self.chunks_changed = 0
        self.chunks_unchanged = 0
        self.chunks_deleted = 0
        self.errors: list[str] = []

    def as_dict(self) -> dict:
        return {
            "entities_processed": self.entities_processed,
            "chunks_total": self.chunks_total,
            "chunks_new": self.chunks_new,
            "chunks_changed": self.chunks_changed,
            "chunks_unchanged": self.chunks_unchanged,
            "chunks_deleted": self.chunks_deleted,
            "errors": self.errors,
        }


def ingest_all(
    since: Optional[datetime] = None,
    entity_types: Optional[list[str]] = None,
    dry_run: bool = False,
) -> IngestionStats:
    """Run a full or incremental ingestion across chats/tasks/notes.

    Args:
        since: Only re-process entities updated after this timestamp.
            None means full reindex.
        entity_types: Restrict to a subset, e.g. ["chat"]. Default
            processes all three.
        dry_run: Skip embeddings, OpenSearch writes, and tracking-table
            writes. The chunkers still run and stats are populated, so
            you can verify which chunks *would* be touched without
            consuming OpenAI quota or mutating state.
    """
    stats = IngestionStats()
    entity_types = entity_types or [
        "chat",
        "task",
        "milestone",
        "note",
        "thread_summary",
        "note_summary",
        "todo",
        "conversation",
        "spotlight_answer",
    ]

    if "chat" in entity_types:
        log.info("Ingesting chats (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_all_chat_chunks(since=since), stats, dry_run=dry_run)
    if "task" in entity_types:
        log.info("Ingesting tasks (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_task_chunks(since=since), stats, dry_run=dry_run)
    if "milestone" in entity_types:
        log.info("Ingesting milestones (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_milestone_chunks(since=since), stats, dry_run=dry_run)
    if "note" in entity_types:
        log.info("Ingesting notes (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_all_note_chunks(since=since), stats, dry_run=dry_run)
    if "thread_summary" in entity_types:
        log.info("Ingesting thread summaries (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_thread_summary_chunks(since=since), stats, dry_run=dry_run)
    if "note_summary" in entity_types:
        log.info("Ingesting note summaries (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_note_summary_chunks(since=since), stats, dry_run=dry_run)
    if "todo" in entity_types:
        log.info("Ingesting todos (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_todo_chunks(since=since), stats, dry_run=dry_run)
    if "conversation" in entity_types:
        log.info("Ingesting conversations (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_conversation_chunks(since=since), stats, dry_run=dry_run)
    if "spotlight_answer" in entity_types:
        log.info("Ingesting spotlight answers (since=%s, dry_run=%s)...", since, dry_run)
        _ingest_stream(iter_spotlight_answer_chunks(since=since), stats, dry_run=dry_run)

    # Refresh-deferred bulk: with `RAG_BULK_REFRESH=false` (the default),
    # individual `_bulk()` calls skip server-side refresh, leaving the
    # written segments invisible to search until we call refresh here.
    # One refresh at the end of the run beats ~N refreshes at one per
    # entity batch — both for throughput AND for total wall-clock.
    # Skipped in dry_run since nothing was written.
    if not dry_run and not settings.SEARCH_ENGINE.get("RAG_BULK_REFRESH", False):
        try:
            client = get_client()
            client.indices.refresh(index=get_index_alias())
            log.info("Refresh complete (deferred-refresh mode).")
        except Exception:  # noqa: BLE001 — refresh failure is non-fatal
            log.exception(
                "Final refresh failed; documents will become searchable on next refresh."
            )

    log.info("Ingestion done: %s", stats.as_dict())
    return stats


def ingest_conversation_run(run) -> bool:
    """Index ONE completed `AgentRun` into the conversation lane,
    immediately searchable (C1 near-real-time hook, §4.7).

    Called by `agent_views._stream_ndjson` right after a run is saved
    with `status="done"` — off the user-visible stream (the last byte
    has already been sent), so the ~1 embed call + bulk write + refresh
    it costs never delays an answer. Returns True when a chunk was
    indexed, False when the run isn't durable memory (wrong status,
    empty answer, missing scope).

    Idempotent with the 10-minute incremental reindexer that remains
    the backstop: `_ingest_entity` hash-diffs against `RagChunk`, so
    when the cron re-sees this run the chunk is `unchanged` and no
    re-embed happens. The explicit refresh matters because the default
    deferred-refresh mode (`RAG_BULK_REFRESH=false`) leaves bulk writes
    invisible to search until *some* refresh runs — without it the
    "recall what we discussed a minute ago" case would silently wait
    for the next cron pass.
    """
    entity = conversation_chunks_for_run(run)
    if entity is None:
        return False
    stats = IngestionStats()
    _ingest_entity(entity, stats, dry_run=False)
    if not settings.SEARCH_ENGINE.get("RAG_BULK_REFRESH", False):
        try:
            get_client().indices.refresh(index=get_index_alias())
        except Exception:  # noqa: BLE001 — refresh failure is non-fatal
            log.exception(
                "Post-conversation refresh failed; the chunk becomes searchable on next refresh."
            )
    return True


def _ingest_stream(
    stream: Iterable[EntityChunks], stats: IngestionStats, *, dry_run: bool
) -> None:
    for entity in stream:
        try:
            _ingest_entity(entity, stats, dry_run=dry_run)
            stats.entities_processed += 1
        except Exception as e:  # noqa: BLE001 — keep one bad entity from killing run
            log.exception("Failed to ingest entity %s/%s", entity.entity_type, entity.entity_id)
            stats.errors.append(f"{entity.entity_type}/{entity.entity_id}: {e}")


def _ingest_entity(entity: EntityChunks, stats: IngestionStats, *, dry_run: bool) -> None:
    if not entity.chunks:
        # Nothing to index, but we may still need to delete stale rows
        # that existed under this entity previously.
        if not dry_run:
            _delete_stale(entity.entity_type, entity.entity_id, keep_chunk_ids=set(), stats=stats)
        return

    model = get_active_embedding_model_name()

    # Compute hashes.
    incoming: dict[str, Chunk] = {}
    incoming_hashes: dict[str, str] = {}
    for c in entity.chunks:
        incoming[c.chunk_id] = c
        incoming_hashes[c.chunk_id] = hash_text(c.search_text or "")

    # Look up existing tracking rows for this entity.
    existing_rows = {
        row.chunk_id: row
        for row in RagChunk.objects.filter(
            entity_type=entity.entity_type, entity_id=entity.entity_id
        )
    }

    new_ids: list[str] = []
    changed_ids: list[str] = []
    unchanged_ids: list[str] = []

    for cid, c in incoming.items():
        row = existing_rows.get(cid)
        if row is None:
            new_ids.append(cid)
        elif row.text_hash != incoming_hashes[cid] or row.embedding_model != model:
            changed_ids.append(cid)
        else:
            unchanged_ids.append(cid)

    to_embed_ids = new_ids + changed_ids
    stats.chunks_total += len(incoming)
    stats.chunks_new += len(new_ids)
    stats.chunks_changed += len(changed_ids)
    stats.chunks_unchanged += len(unchanged_ids)

    if dry_run:
        # Count stale rows for stats, but don't touch OpenSearch or
        # the tracking table.
        keep_ids = set(incoming.keys())
        stale_count = (
            RagChunk.objects.filter(entity_type=entity.entity_type, entity_id=entity.entity_id)
            .exclude(chunk_id__in=keep_ids)
            .count()
        )
        stats.chunks_deleted += stale_count
        return

    # Embed new + changed.
    embeddings_by_id: dict[str, list[float]] = {}
    if to_embed_ids:
        texts = [incoming[cid].search_text for cid in to_embed_ids]
        vectors = embed_texts(texts)
        for cid, vec in zip(to_embed_ids, vectors):
            embeddings_by_id[cid] = vec

    # Bulk index new + changed.
    if to_embed_ids:
        _bulk_index(
            [_to_os_doc(incoming[cid], embeddings_by_id[cid], model) for cid in to_embed_ids]
        )

    # Stale = previously tracked under this entity but not in current set.
    keep_ids = set(incoming.keys())
    _delete_stale(entity.entity_type, entity.entity_id, keep_chunk_ids=keep_ids, stats=stats)

    # Mirror to RagChunk.
    _upsert_tracking(
        entity_type=entity.entity_type,
        entity_id=entity.entity_id,
        chunks=incoming,
        hashes=incoming_hashes,
        model=model,
        affected_ids=set(to_embed_ids),
    )


def _to_os_doc(chunk: Chunk, embedding: list[float], model: str) -> dict:
    """Build the OpenSearch document for a chunk.

    The chunk_id is also used as the OpenSearch `_id` so that upsert
    semantics work naturally.
    """
    doc = chunk.to_dict()
    doc["embedding"] = embedding
    doc["embedding_model"] = model
    doc["index_schema_version"] = INDEX_SCHEMA_VERSION
    doc["text_hash"] = hash_text(chunk.search_text or "")
    return {
        "_index": get_index_alias(),
        "_id": chunk.chunk_id,
        "_source": doc,
    }


def _bulk_index(actions: list[dict]) -> None:
    if not actions:
        return
    batch_size = settings.SEARCH_ENGINE["BULK_BATCH_SIZE"]
    client = get_client()
    # `raise_on_error=False` so a single bad doc doesn't abort the
    # whole batch; we log instead.
    #
    # `refresh` strategy: default `RAG_BULK_REFRESH=false` means each
    # batch ships fire-and-forget (no `?refresh`), and `ingest_all`
    # issues one explicit `indices.refresh()` at the end of the full
    # run. Switching to `true` triggers per-batch refresh — useful for
    # one-off writes that need to be visible immediately (e.g. an
    # ad-hoc `manage.py shell` upsert).
    refresh_per_batch = settings.SEARCH_ENGINE.get("RAG_BULK_REFRESH", False)
    success, errors = os_helpers.bulk(
        client,
        actions,
        chunk_size=batch_size,
        raise_on_error=False,
        raise_on_exception=False,
        refresh=bool(refresh_per_batch),
    )
    if errors:
        # ERROR (not WARNING) so a CronCommand-based reindex fails the run —
        # a bulk that wrote nothing must not leave the cron green.
        log.error("Bulk index reported %d errors (success=%d)", len(errors), success)
        for err in errors[:5]:
            log.warning("  %s", err)


def _delete_stale(
    entity_type: str,
    entity_id: str,
    keep_chunk_ids: set[str],
    stats: IngestionStats,
) -> None:
    """Delete chunks tracked for this entity but missing from the
    regeneration. Both OpenSearch and RagChunk are cleaned."""
    stale_ids = list(
        RagChunk.objects.filter(entity_type=entity_type, entity_id=entity_id)
        .exclude(chunk_id__in=keep_chunk_ids)
        .values_list("chunk_id", flat=True)
    )
    if not stale_ids:
        return

    actions = [
        {"_op_type": "delete", "_index": get_index_alias(), "_id": cid} for cid in stale_ids
    ]
    client = get_client()
    # Same deferred-refresh policy as `_bulk_index` — defers to the
    # end-of-run refresh in `ingest_all` unless RAG_BULK_REFRESH=true.
    refresh_per_batch = settings.SEARCH_ENGINE.get("RAG_BULK_REFRESH", False)
    _, del_errors = os_helpers.bulk(
        client,
        actions,
        raise_on_error=False,
        raise_on_exception=False,
        refresh=bool(refresh_per_batch),
    )
    if del_errors:
        log.error("Stale-chunk delete reported %d errors", len(del_errors))
    RagChunk.objects.filter(chunk_id__in=stale_ids).delete()
    stats.chunks_deleted += len(stale_ids)


@transaction.atomic
def _upsert_tracking(
    *,
    entity_type: str,
    entity_id: str,
    chunks: dict[str, Chunk],
    hashes: dict[str, str],
    model: str,
    affected_ids: set[str],
) -> None:
    """Insert or update RagChunk rows for the chunks we just indexed.

    `affected_ids` are the new+changed ids (got re-embedded); we
    update their hash + indexed_at. Unchanged rows just get their
    indexed_at refreshed implicitly via the next reindex; we leave
    them alone here to avoid pointless writes.
    """
    for cid in affected_ids:
        c = chunks[cid]
        RagChunk.objects.update_or_create(
            chunk_id=cid,
            defaults={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "chunk_type": c.chunk_type,
                "team_id": c.team_id,
                "text_hash": hashes[cid],
                "embedding_model": model,
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "source_version": None,
            },
        )
