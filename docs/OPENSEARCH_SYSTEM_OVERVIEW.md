# OpenSearch system — recap & reference

A single-pager reference for how the OpenSearch indexing + search layer
works end-to-end. Use this when you need to remember: *what's in the
index, how does a query get answered, where do I tune X*. Phased
implementation history lives in `01_...md` through `14_...md`; this
doc is the steady-state snapshot.

Last refreshed for **`INDEX_SCHEMA_VERSION = "v2"`** (see
[index_config.py](../backend_django/origin/search_engine/index_config.py)).

---

## TL;DR

- One OpenSearch index holds **chunks** from chats, tasks, notes,
  thread summaries, and note summaries. Each chunk has a vector
  embedding + keyword fields + tenant/ACL metadata.
- Two consumers query it: **Spotlight Cmd-K** (`/api/v2/search/`) and
  the agent's **`search_knowledge_base`** tool. Both route through the
  same `search()` function with different `mode=` settings.
- Search is **hybrid**: BM25 (keyword lane) + k-NN (vector lane), fused
  via **Reciprocal Rank Fusion (RRF)**, then chunk-type re-weighted,
  freshness-decayed, grouped per entity, and (optionally) reranked.
- Thread Q&A and Note Q&A endpoints are **DB-only** — they read note /
  thread rows directly. They don't hit OpenSearch.
- Writes go through the **chunker → ingestion pipeline** — diff against
  `RagChunk` tracking table, embed only the new/changed chunks, bulk-
  index, delete stale. Idempotent.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Postgres (source of truth)                                       │
│  ChatMaster / DMMessages / TaskMaster / Note*Master / ...        │
│         │                                                         │
│         │ python manage.py opensearch_reindex                     │
│         ▼                                                         │
│  ┌───────────────┐    ┌────────────────────┐                     │
│  │   Chunkers    │ →  │  ingestion.py      │                     │
│  │ (5 modules)   │    │  diff/embed/upsert │                     │
│  └───────────────┘    └────────────────────┘                     │
│         │                       │                                 │
│         │ EntityChunks          │ embed_texts() (OpenAI/Vertex)  │
│         ▼                       ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  OpenSearch index (alias: knowledge_chunks_current)          │ │
│  │   ─ chunks with embedding + keyword + text fields            │ │
│  │   ─ HNSW (cosine, Lucene)                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ▲                                                         │
│         │ hybrid query                                            │
│  ┌──────┴──────────────┐                                          │
│  │  search.py          │ ← Spotlight UI (/api/v2/search/)         │
│  │  hybrid pipeline    │ ← Agent tool (search_knowledge_base)     │
│  └─────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## What's indexed

Each chunker produces `EntityChunks` (a batch of chunks for one
logical entity). The ingestion pipeline diffs incoming chunks against
the `RagChunk` tracking table by `text_hash` and only re-embeds the
new/changed ones.

| Chunker | Entity type | Chunk type(s) | When | Notes |
|---|---|---|---|---|
| [chat_chunker.py](../backend_django/origin/search_engine/chunkers/chat_chunker.py) | `chat` | `chat_message`, `chat_thread_window` | DM/GM/PM/MDM messages, threads | Focal message + N preceding messages folded into `search_text` for context (`RAG_CHAT_CONTEXT_WINDOW=2`). Window chunks suppressed when a `ThreadSummary` row exists. |
| [task_chunker.py](../backend_django/origin/search_engine/chunkers/task_chunker.py) | `task` | `task_title_content`, `task_content_chunk` (long bodies), `task_comment` | TaskMaster + TaskComments | Long bodies (>1500 chars) split: title chunk + content chunk. Comments get one chunk each. All carry `task_status`/`assignee_id`/etc. |
| [note_chunker.py](../backend_django/origin/search_engine/chunkers/note_chunker.py) | `note` | `note_section` | PersonalNote / TaskNote / ChatNote | Heading-bounded sections. Title prepended to every section's `search_text` so topical queries still surface the right section. |
| [thread_summary_chunker.py](../backend_django/origin/search_engine/chunkers/thread_summary_chunker.py) | `thread_summary` | `thread_summary` | `ThreadSummary` rows (created on-demand when a user clicks ✨ Ask) | One chunk per summary. |
| [note_summary_chunker.py](../backend_django/origin/search_engine/chunkers/note_summary_chunker.py) | `note_summary` | `note_summary` | `NoteSummary` rows (created on-demand) | One chunk per summary. |

**Why overlapping coverage** (e.g. a chat message + the thread window +
a thread summary all referencing the same content): each captures a
different granularity. Keyword search benefits from individual
messages; semantic search benefits from thread-level or LLM-curated
abstracts. Overlap is handled at query time via `RAG_DEDUP_BY_HASH` +
chunk-type re-weighting (`_MODE_CONFIG[mode].chunk_type_weights`).

---

## Index schema (v2)

Defined in [index_config.py](../backend_django/origin/search_engine/index_config.py).
Mappings, by purpose:

**Identity / grouping** — used for filtering, deletion, and entity
roll-up at the response layer:

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | keyword | PK; also the OpenSearch `_id` |
| `entity_type` | keyword | `chat` / `task` / `note` / `thread_summary` / `note_summary` |
| `entity_id` | keyword | Grouping key — `dm:5:thread:3`, `task:42`, `note:chat:18`, etc. |
| `chunk_type` | keyword | `chat_message`, `task_title_content`, `note_section`, etc. — drives chunk-type re-weighting |

**Tenant / ACL** — applied as `filter` clauses (no scoring impact):

| Field | Type | Notes |
|---|---|---|
| `team_id` | keyword | Mandatory tenant boundary |
| `acl_user_ids` | keyword (multi) | Denormalized — query filters by `terms` on the requesting user |

**Type-specific identifiers** — populated by the chunker that owns the
entity; nullable on other chunk types:

| Field | Set by |
|---|---|
| `chat_type`, `chat_id`, `thread_id` | chat / note chunkers (when note attached to chat) |
| `task_id` | task chunker / task-note chunker |
| `note_id`, `note_type` | note / note-summary chunkers |
| `project_id` | task / PM-chat / task-note chunkers |

**v2 overlays** — added so hybrid search can filter "what did Bob
say", "my open tasks about X", "my notes about Y" without DB round-trips:

| Field | Set by | Unlocks |
|---|---|---|
| `author_id`, `author_name`, `chat_message_id` | chat focal-message chunks | "What did Bob say about X" filter; source-chip rendering without DB |
| `task_status`, `task_priority`, `task_assignee_id`, `task_milestone_id`, `task_sprint_id` | task chunker (incl. comments) | Status/priority/assignee filters in hybrid search |
| `note_owner_id`, `note_parent_id` | note + note-summary chunkers | "My notes about X"; parent-note traversal |

**Searchable text fields**:

| Field | Type | Analyzer | Notes |
|---|---|---|---|
| `title` | text | standard (+ `.prefix` edge n-gram subfield) | Boosted `^3` in BM25; `.prefix^4` for typeahead |
| `search_text` | text | standard (+ `.en` English-stemmer subfield) | Full chunk body for embedding + BM25; `.en` recovers conjugation variants |
| `snippet_text` | text | standard | Short UI excerpt; boosted `^2` |

**Vector**:

| Field | Type | Notes |
|---|---|---|
| `embedding` | knn_vector (1536-dim, HNSW, cosine, Lucene engine) | Always excluded from `_source` projections |

**Cross-entity / housekeeping**:

| Field | Type | Notes |
|---|---|---|
| `related_entity_ids` | keyword (multi) | Task→chat link, note→parent-note, etc. Surfaced in source chips as a deep-link fallback |
| `created_at` / `updated_at` | date | Date-range filters + freshness decay |
| `text_hash` | keyword | SHA-256 of `search_text`; dedup at query time |
| `embedding_model` | keyword | Recorded so a provider/model swap triggers re-embed |
| `index_schema_version` | keyword | Currently `"v2"`; the reindex command warns on mismatch |

**Index-level settings** (configurable via env so production can tune
without code changes):

| Setting | Env var | MVP default |
|---|---|---|
| `number_of_shards` | `OPENSEARCH_SHARDS` | 1 |
| `number_of_replicas` | `OPENSEARCH_REPLICAS` | 0 |
| `refresh_interval` | `OPENSEARCH_REFRESH_INTERVAL` | `1s` |
| `index.knn` | (fixed) | true |

---

## Search pipeline

Owner: [search.py](../backend_django/origin/search_engine/search.py).

```
search(query=..., mode=...)
  │
  ├─ build filter: team_id + acl_user_ids + optional entity_types + date range
  │
  ├─ (optional) query rewriting → N variants  [RAG_USE_QUERY_REWRITE]
  │
  ├─ for each variant:
  │    ├─ _run_keyword  → BM25 multi_match over title^3 / title.prefix^4 /
  │    │                  snippet_text^2 / search_text / search_text.en^0.8
  │    └─ _run_vector   → k-NN over embedding, per-mode ef_search
  │
  ├─ _rrf_fuse(keyword_hits, vector_hits)   per variant
  ├─ weighted-sum across variants            (original=2.0, others=1.0)
  ├─ chunk-type re-weight                    (per-mode weights)
  ├─ freshness decay  exp(-age / half_life)  [RAG_FRESHNESS_HALF_LIFE_DAYS=90]
  ├─ text-hash dedup                         [RAG_DEDUP_BY_HASH=true]
  ├─ group by entity                         (keep highest score per entity)
  ├─ relevance threshold                     (absolute floor + ratio)
  ├─ representation floor                    (optional, off)
  ├─ (optional) Cohere/LLM reranker          [RAG_USE_RERANKER=false]
  └─ apply friendly chat titles → return
```

### Search modes

Set via the `mode=` kwarg. Picks a hyperparameter set from `_MODE_CONFIG`:

| Mode | Caller | `pool_size` | `ef_search` | Freshness | Chunk-type weights |
|---|---|---|---|---|---|
| `typeahead` | Spotlight UI (default when `for_agent=False`) | 20 | 64 | on | **Precision-favoured** — raw messages/sections/task titles outrank LLM summaries |
| `ai_search` | `search_knowledge_base` tool (default when `for_agent=True`) | 60 | 128 | on | **Recall-favoured** — `thread_summary` / `note_summary` ranked higher (LLM reads abstracts better) |
| `eval` | `agent_eval --retrieval` harness | 100 | 256 | off | **Flat** — retrieval-quality numbers reflect raw BM25 + vector + RRF |

### Reciprocal Rank Fusion (RRF)

Score for chunk `c` is `Σ 1 / (RRF_K + rank_c)` across all lanes that
ranked it. `RRF_K=60`. Each lane is bounded to a `pool_size` of
candidates. Chunks ranked by ONE lane only still pick up some score —
hybrid degrades gracefully if e.g. the embedding API is down.

### Chunk-type re-weighting (v2)

Applied after RRF, multiplicative on the score. Lets typeahead favour
literal-keyword hits and ai_search favour LLM-curated abstracts without
changing the schema or requiring a reindex. See `_MODE_CONFIG` in
search.py:51-110.

### Freshness boost

`score *= exp(-age_days / half_life)`. Default `half_life=90` days.
Disabled by setting `RAG_FRESHNESS_HALF_LIFE_DAYS=0`. Off in
`mode="eval"` to keep retrieval-quality numbers deterministic.

### Result shape

The Spotlight UI shape returns one row per entity with a snippet and
metadata. The agent's `search_knowledge_base` path (`for_agent=True`)
gets the same shape plus up to `max_chunks_per_entity` full chunk
bodies (`search_text`) so the LLM has grounding context.

---

## ACL & tenant isolation

Every chunk carries `team_id` + `acl_user_ids`. Every search query
filters on:

- `term: {team_id: <requesting team>}`
- `term: {acl_user_ids: <requesting user>}`

Both are in the `filter` context (no scoring impact). ACL is
**denormalized** — the chunker resolves the allowed user set once and
stamps it on each chunk, so retrieval is a single `terms` clause with
no joins.

ACL helpers per entity type live in
[agent/acl.py](../backend_django/origin/search_engine/agent/acl.py):

- `chat_acl_user_ids(chat_type, chat_id)` — DM members / GM members /
  PM members / MDM members
- `task_acl_user_ids(project_id, assignee_id, reporter_id)` — project
  members + assignee + reporter
- `personal_note_acl_user_ids(...)`, `task_note_acl_user_ids(...)`,
  `chat_note_acl_user_ids(...)` — owner + parent-context members +
  explicit `NotePermissionMaster` grants

**Permission changes** (adding a member to a chat, etc.) require
re-ingesting the affected entity so the new `acl_user_ids` lands in
OpenSearch. Today this happens via the periodic `opensearch_reindex`
cron rather than instantly on the event.

---

## Embedding pipeline

Owner: [embeddings/](../backend_django/origin/search_engine/embeddings/).

- **Provider** — `EMBEDDING_PROVIDER=openai` (default, `text-embedding-3-small`, 1536-dim) or `vertex` (`gemini-embedding-001`, truncated to 1536-dim via Matryoshka so we can swap providers without re-creating the index).
- **Batching** — `SEARCH_EMBEDDING_BATCH_SIZE=100` items per provider request.
- **Caching** — two-tier:
  - **L1**: per-worker `lru_cache` (256 entries, model+text key). Survives keystroke bursts in Spotlight typeahead.
  - **L2**: Redis-backed, TTL `RAG_EMBEDDING_CACHE_TTL_S=600s`. Shared across workers, survives restart.
- **Asymmetric encoding** — Vertex distinguishes "document" vs "query" task types. OpenAI ignores it.
- **Retry** — up to 3 attempts with exponential backoff on transient errors.

**Cost model**: only new/changed chunks (different `text_hash`) get
embedded on each reindex. Unchanged chunks pay zero. Switching
embedding model (different `embedding_model` recorded in `RagChunk`)
forces a full re-embed.

---

## Ingestion / write path

Owner: [ingestion.py](../backend_django/origin/search_engine/ingestion.py).

Triggered by:

- `python manage.py opensearch_reindex` — full or incremental (via
  `--since-minutes` / `--since`)
- Demo signin handler — kicks off an incremental reindex so the demo
  workspace's content surfaces in Spotlight within ~12 s

Per-entity flow:

1. Chunker yields `EntityChunks(entity_type, entity_id, chunks=[...])`
2. Compute `text_hash` per chunk
3. Diff against `RagChunk` rows for this `entity_id`:
   - **new** — never seen → embed + index
   - **changed** — same chunk_id, different text_hash or model → re-embed + index
   - **unchanged** — skip
   - **stale** — present in RagChunk but missing from this run → delete
4. Bulk-index new+changed (one `_bulk` call per entity batch)
5. Bulk-delete stale
6. Mirror to `RagChunk` so the next run can repeat steps 1-3

**Refresh policy (v2)**: with `RAG_BULK_REFRESH=false` (default), each
`_bulk()` ships without `?refresh`, and `ingest_all` issues one
explicit `indices.refresh()` at the end of the full run. Saves ~1s
per batch. Switch to `true` for one-off writes that need to be
searchable immediately.

---

## Operational guide

### Routine commands

```bash
# Create the index (idempotent — leaves existing alone)
python manage.py opensearch_setup

# Recreate the index (DESTRUCTIVE — required when schema changes)
python manage.py opensearch_setup --recreate

# Full reindex
python manage.py opensearch_reindex

# Incremental reindex (suitable for crontab; the demo-signin handler also calls this)
python manage.py opensearch_reindex --since-minutes 10

# Reindex one entity type only
python manage.py opensearch_reindex --entity-types task

# Dry-run (no embeddings, no writes — just counts)
python manage.py opensearch_reindex --dry-run
```

### Schema migration

Bump `INDEX_SCHEMA_VERSION` in
[index_config.py](../backend_django/origin/search_engine/index_config.py)
whenever you change the mapping (added/removed fields, analyzer
changes, embedding dim changes). Then:

```bash
python manage.py opensearch_setup --recreate
python manage.py opensearch_reindex
```

The `opensearch_reindex` command samples one chunk's
`index_schema_version` and warns if it doesn't match the code's
expected value.

### Environment variables (subset — see settings.py)

| Env var | Default | Effect |
|---|---|---|
| `OPENSEARCH_HOST` / `OPENSEARCH_PORT` | `opensearch` / 9200 | Cluster endpoint |
| `OPENSEARCH_INDEX` / `OPENSEARCH_ALIAS` | `knowledge_chunks_v1` / `knowledge_chunks_current` | Physical + alias names |
| `OPENSEARCH_SHARDS` | `1` | Shard count (single-node MVP) |
| `OPENSEARCH_REPLICAS` | `0` | Replica count |
| `OPENSEARCH_REFRESH_INTERVAL` | `1s` | OS refresh cadence |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `vertex` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536-dim |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Must match mapping |
| `SEARCH_EMBEDDING_BATCH_SIZE` | `100` | Per provider request |
| `SEARCH_BULK_BATCH_SIZE` | `200` | Per `_bulk` call |
| `RAG_BULK_REFRESH` | `false` | If true, refresh per batch (vs once at end of run) |
| `RAG_CHAT_CONTEXT_WINDOW` | `2` | Preceding messages folded into chat_message search_text |
| `RAG_BM25_TITLE_BOOST` / `RAG_BM25_SNIPPET_BOOST` | `3` / `2` | Field-level BM25 boosts |
| `RAG_FRESHNESS_HALF_LIFE_DAYS` | `90` | Freshness decay half-life; 0 disables |
| `RAG_DEDUP_BY_HASH` | `true` | Drop same-text duplicates |
| `RAG_MIN_SCORE` / `RAG_MIN_SCORE_RATIO` | `0.040` / `0.5` | Relevance threshold |
| `RAG_USE_QUERY_REWRITE` | `false` | Expand query into N variants via LLM (agent path only) |
| `RAG_REWRITE_NUM_VARIANTS` / `RAG_REWRITE_ORIGINAL_WEIGHT` | `3` / `2.0` | Rewriting tuning |
| `RAG_USE_RERANKER` | `false` | Post-hoc Cohere or LLM reranker |
| `RAG_RERANKER_PROVIDER` | `llm` | `llm` or `cohere` |
| `RAG_EMBEDDING_CACHE_TTL_S` | `600` | L2 Redis embedding cache TTL |

---

## Performance levers (v2)

Free-throughput wins shipped in the v2 cycle. None require a reindex
once the schema is in place.

| Knob | Lever | Effect |
|---|---|---|
| HNSW `ef_search` | Per-query in `_run_vector` (64 typeahead / 128 ai_search / 256 eval) | Drops Lucene default of 512; halves vector-query latency |
| `track_total_hits: false` | Both lanes | Skips exact total-count aggregation |
| `_source.excludes: [embedding]` | Both lanes | Vector blob never returned over the wire |
| Refresh deferral | `RAG_BULK_REFRESH=false` + end-of-run refresh | One refresh per full reindex, not N per batch |
| Embedding L2 cache | Redis-backed, shared | Cross-worker hits + survives restart |
| Chunk-type re-weighting | Per-mode weights, applied post-RRF | Mode-appropriate ranking without schema change |

---

## File map

What lives where, in [backend_django/origin/search_engine/](../backend_django/origin/search_engine/):

```
search_engine/
├── chunkers/
│   ├── base.py                    Chunk + EntityChunks dataclasses
│   ├── chat_chunker.py            DM/GM/PM/MDM → chat_message, chat_thread_window
│   ├── task_chunker.py            TaskMaster → task_title_content, task_content_chunk, task_comment
│   ├── note_chunker.py            Personal/Task/ChatNote → note_section
│   ├── thread_summary_chunker.py  ThreadSummary → thread_summary
│   └── note_summary_chunker.py    NoteSummary → note_summary
├── embeddings/
│   ├── __init__.py                embed_one / embed_texts / hash_text / caching
│   ├── openai_embedder.py         OpenAI provider
│   └── vertex_embedder.py         Google Vertex provider
├── agent/
│   ├── acl.py                     Per-entity-type ACL helpers (shared between chunkers + tool ACL)
│   ├── controller.py              Agent loop, source-chip resolution, citation handling
│   ├── tools/
│   │   ├── search_kb.py           The ONLY agent tool that hits OpenSearch
│   │   ├── fetch_task.py / fetch_note.py / ...   DB-only tools
│   │   └── ...
│   ├── thread_summary.py          LLM thread summarization
│   ├── note_summary.py            LLM note summarization
│   ├── citation_resolver.py       Resolve unresolved [type:id] citations in final answer
│   └── evals/runner.py            agent_eval --retrieval harness (uses mode="eval")
├── management/commands/
│   ├── opensearch_setup.py        Create / recreate the index
│   └── opensearch_reindex.py      Run the ingestion pipeline
├── ingestion.py                   Orchestrator — diff/embed/upsert
├── index_config.py                Schema + index settings; INDEX_SCHEMA_VERSION lives here
├── search.py                      Hybrid retrieval pipeline
├── views.py                       /api/v2/search/ endpoint (Spotlight backend)
├── agent_views.py                 /agent/ask/, /agent/thread-summary/, /agent/note-summary/
├── models.py                      RagChunk tracking + ThreadSummary + NoteSummary + AgentSession
├── text_extraction.py             BlockNote → plain text + heading-based sections
├── friendly_titles.py             Viewer-aware chat-name resolution at query time
├── opensearch_client.py           Singleton opensearchpy client + alias helpers
└── reranker.py                    Optional Cohere/LLM reranker (flag-gated)
```

---

## Common workflows

### "I changed the schema — what now?"

1. Bump `INDEX_SCHEMA_VERSION` in `index_config.py`
2. `python manage.py opensearch_setup --recreate`
3. `python manage.py opensearch_reindex`
4. (Optional) `python manage.py agent_eval --retrieval` to confirm no regression (fast, no LLM cost)

### "Search results look stale"

Check refresh policy. With `RAG_BULK_REFRESH=false`, writes are not
visible until the end-of-run refresh (or the next `OPENSEARCH_REFRESH_INTERVAL`
tick, default 1s). Run `python manage.py opensearch_reindex` or set
`RAG_BULK_REFRESH=true` for the immediate-visibility path.

### "A user can't find a chat they're in"

Likely an ACL update that hasn't been re-ingested. Check:

```python
client.get(index=alias, id="chat:dm:5:msg:42")["_source"]["acl_user_ids"]
```

If the user's id isn't in the list, force a reindex of the affected
entity. Long-term fix: wire socket events to incremental reindex on
permission changes.

### "Adding a new entity type"

1. Add a chunker in `chunkers/` that yields `EntityChunks` with a new
   `entity_type` value.
2. Add the chunker to the entity-type list in `ingestion.py`
   (`ingest_all` and `_ingest_stream`).
3. Add the new type to the `--entity-types` choices in
   `management/commands/opensearch_reindex.py`.
4. Add ACL resolution logic — usually a new helper in `agent/acl.py`
   following the existing patterns.
5. If query-side ranking should treat the new chunk type specially,
   add weights to each entry in `_MODE_CONFIG` in `search.py`.
6. Bump `INDEX_SCHEMA_VERSION` if you added new top-level fields;
   otherwise the existing mapping accepts the new chunks unchanged.

### "Debugging a missing result"

1. Confirm the chunk was indexed: `client.search(index=alias, body={"query": {"term": {"entity_id": "..."}}})`
2. Check the chunk's `acl_user_ids` includes the user
3. Check the chunk's `text_hash` matches what's in `RagChunk` (mismatch
   means the index has a different version of the chunk than the
   tracking table; usually a stale row that survived a partial
   reindex — run a full reindex to reconcile).
4. Run the same query in `mode="eval"` (flat weights, no freshness,
   wider pool) to rule out scoring/ranking issues vs presence in the
   index.

### "Working on retrieval quality"

The eval harness in
[agent/evals/](../backend_django/origin/search_engine/agent/evals/)
is the canonical regression suite. Two suites share one CLI:

- **Retrieval** (`--retrieval`) — direct `search(...)` calls, no LLM, asserts on the ranked entity list. Source: `agent/evals/retrieval_cases.yaml`. Fast.
- **Behavior** (default) — full agent loop with real Gemini/Claude calls. Source: `agent/evals/cases.yaml`. Slower + costs LLM credits.

```bash
python manage.py agent_eval --retrieval            # retrieval suite (no LLM)
python manage.py agent_eval                        # behavior suite (full loop)
python manage.py agent_eval --all                  # both
python manage.py agent_eval --case <id>            # one case
```

Acceptance for OpenSearch changes: the retrieval suite passes (recall@10 + MRR don't regress).

---

## What doesn't hit OpenSearch

- **Thread Q&A** (`/agent/thread-summary/`) — DB-only. Reads
  DM/GM/PM/MDM message tables directly.
- **Note Q&A** (`/agent/note-summary/`) — DB-only. Reads
  Personal/Task/ChatNote tables directly.
- **Most agent tools** — `fetch_task`, `fetch_note`,
  `fetch_chat_thread`, `list_tasks`, `list_projects`, write tools, etc.
  Only `search_knowledge_base` queries OpenSearch.
- **Frontend deep-links / navigation** — pure URL routing.

The summary endpoints DO write back into OpenSearch via
`thread_summary_chunker` / `note_summary_chunker` on the next ingest
cycle — so their abstracts become discoverable through Spotlight.

---

## Related docs

- [OPENSEARCH_COMMANDS.md](OPENSEARCH_COMMANDS.md) — handy command cheat sheet
- [01_agentic_ai_opensearch_mvp_roadmap.md](../mvp_roadmap/01_agentic_ai_opensearch_mvp_roadmap.md) — original roadmap
- [02_opensearch_mvp_search_engine_design.md](../mvp_roadmap/02_opensearch_mvp_search_engine_design.md) — initial design
- [03_opensearch_mvp_implementation.md](../mvp_roadmap/03_opensearch_mvp_implementation.md) — initial impl notes
- [appendix_opensearch_hybrid_search_guide.md](appendix_opensearch_hybrid_search_guide.md) — RRF + hybrid-search deep-dive
- [021_opensearch_chat_rag_chunking_summary.md](../mvp_roadmap/021_opensearch_chat_rag_chunking_summary.md) — chat chunking rationale
- [SPOTLIGHT_DEMO_VIDEO_SCRIPT.md](../spotlight/SPOTLIGHT_DEMO_VIDEO_SCRIPT.md) — the actual demo-day prompts; ground-truth use cases for retrieval quality
