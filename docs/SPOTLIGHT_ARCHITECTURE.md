# Spotlight (Genos AI) — Architecture Overview

A holistic map of how the Cmd-K Spotlight overlay works end-to-end: what runs in the browser, what runs on the Django backend, how OpenSearch is used, where the LLM fits in, and how everything gets stitched together. The goal is to give you the mental model you need to make changes confidently — for per-phase implementation depth see the [companion phase docs](#9-where-to-dig-deeper).

---

## 1. What Spotlight is

Cmd-K (or Ctrl-K on non-mac) opens a translucent overlay that does two things in one input box:

- **Pure search (typeahead)** — every keystroke fires a debounced cross-workspace search across chats, tasks, notes, and collected past AI answers (a question the team has already asked resurfaces instantly — see §7 *Answer reuse*). No LLM, no cost, sub-300ms.
- **AI ask (agent)** — press Enter and the same input becomes a question to Genos AI. The backend runs a multi-step tool-using LLM loop, streams events back, and renders a cited answer with clickable inline citations.

The overlay lives in [SpotlightOverlay.tsx](../genos-frontend/src/features/spotlight/SpotlightOverlay.tsx); its state machine and SSE plumbing live in [useSpotlight.ts](../genos-frontend/src/features/spotlight/useSpotlight.ts).

---

## 2. The 30-second mental model

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (React)                                 │
│                                                                              │
│   SpotlightOverlay   ──>  useSpotlight hook  ──>  agentApi (typed client)    │
│   (rendering)             (state + SSE)           (fetch + NDJSON parser)    │
│                                                                              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │  HTTPS
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DJANGO BACKEND (DRF views)                             │
│                                                                              │
│   GET  /api/v2/search/             SearchView          (typeahead)           │
│   POST /api/v2/agent/ask/          AgentAskView        (streaming NDJSON)    │
│   POST /api/v2/agent/decide/       AgentDecideView     (resume after pause)  │
│   GET  /api/v2/agent/usage/        AgentUsageView      (daily limit)         │
│   GET  /api/v2/agent/sessions/     AgentSessionsListView  (history list)     │
│   GET  /api/v2/agent/sessions/<id>/ AgentSessionDetailView (history detail)  │
│                                                                              │
│   Routed in [search_engine/urls.py]                                          │
│                                                                              │
└──────────────┬────────────────┬───────────────┬────────────────┬─────────────┘
               │                │               │                │
               ▼                ▼               ▼                ▼
       ┌───────────┐    ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ OpenSearch│    │  Postgres  │   │   Redis    │   │ LLM provider│
       │           │    │            │   │            │   │  (Gemini /  │
       │ hybrid    │    │ - app data │   │ - query    │   │   Claude)   │
       │ search    │    │ - Agent*   │   │   embed L2 │   │             │
       │ (BM25 +   │    │   models   │   │ - rewrite  │   │ function-   │
       │  dense)   │    │ - perms    │   │   cache    │   │ calling API │
       └───────────┘    └────────────┘   └────────────┘   └────────────┘
```

There is **no separate service** for the agent — it's a Python function (`run_agent`) called from the view, streaming an iterator back to the client via `StreamingHttpResponse`.

---

## 3. The two request lifecycles

### A. Pure search (typeahead)

```
User types "fram"
   │
   ▼
[useSpotlight.ts] debounces 250ms, fires GET /api/v2/search/?q=fram&team_id=...
   │
   ▼
[search_engine/views.py SearchView]
   │
   ▼
[search_engine/search.py `search()`]  ─── hybrid pipeline:
   │
   ├── _run_keyword(q)   ──> OpenSearch BM25 query (multi_match, title^3, body^2)
   ├── _run_vector(q)    ──> embed q via embed_one() ──> kNN query against
   │                                                     `embedding` field
   │                          (L1 LRU + L2 Redis cache on the query embedding)
   │
   ├── _multi_variant_fuse  ──> if RAG_USE_QUERY_REWRITE: LLM expands `q` into N
   │                            variants; each runs both lanes; results
   │                            RRF-fused with original weighted 2× (default off)
   │
   ├── _rrf_fuse(keyword, vector)  ──> Reciprocal Rank Fusion (k=60)
   ├── _group_by_entity(...)       ──> collapse chunk-level hits into per-entity
   │                                   rows (entity score = max chunk score,
   │                                   keep top 3 chunks per entity)
   ├── _apply_freshness(...)       ──> exponential decay by `updated_at`
   ├── _dedup_by_text_hash(...)    ──> drop near-identical bodies
   ├── _apply_relevance_threshold  ──> drop scores < top_score × ratio
   │                                   (keeps `RAG_THRESHOLD_MIN_SURVIVORS` if
   │                                    threshold would leave too few)
   ├── _apply_representation_floor ──> guarantee ≥1 per entity_type that had any
   │                                   match (flag-gated)
   └── (optional) reranker         ──> LLM or Cohere re-orders top-K (default off)
   │
   ▼
JSON response: list of SpotlightResult dicts (entity_id, title, snippet, score, ...)
   │
   ▼
[SpotlightOverlay] renders results in 3 sections (Chats / Tasks / Notes)
                   with HighlightedText regex over the matched terms.
```

> Collected past AI answers surface in this same typeahead list as a
> `spotlight_answer` lane (typeahead only — deliberately excluded from the
> agent's own grounding). See §7 *Answer reuse* for the lane and its ACL.

**Two latency targets are met today**: typeahead-to-results ~250ms (debounce floor); per-keystroke render is decoupled from search via `useDeferredValue` so the input never stutters.

### B. AI ask (agent loop)

```
User types question, presses Enter
   │
   ▼
[useSpotlight.ts onAsk]  ──> sets ask.isStreaming=true synchronously
                              (this is what flips the layout to "agent mode"
                               with the input row at the bottom — see §4)
   │
   ▼
[agentApi.askAgentStream]  ──> POST /api/v2/agent/ask/
                                body: { query, team_id, session_id?, allow_web_search? }
                                Reads response.body as a ReadableStream and parses
                                line-delimited JSON events (NDJSON).
   │
   ▼
[agent_views.py AgentAskView.post]
   │
   ├── ACL: daily limit check via AgentRun count today (free tier 5/day default,
   │                                                    unlimited via UserFeatureAccess)
   ├── Session: _get_or_create_session(...) — Phase 8 multi-turn memory
   │            (existing or fresh AgentSession, TTL-aware)
   ├── Prior turns: _load_prior_turns(session, SESSION_MAX_PRIOR_TURNS=3)
   │                OR (if RAG_SESSION_ROLLING_SUMMARY=true) load up to 20 and
   │                pass through multi_turn.build_prior_context() which summarises
   │                older ones into a single context note.
   ├── Persistence: create AgentRun row (status="running")
   │
   └── return StreamingHttpResponse(
          _stream_ndjson(
              run_agent(query, ctx, emit_cb, run_id, prior_summary, trace_hook),
              ...
          ),
          content_type="application/x-ndjson",
       )
   │
   ▼
[agent/controller.py run_agent]  ──> wraps _drive_loop with optional self-critique
   │
   ▼
[_drive_loop]  ──> the heart of the agent (see §5 for full state diagram)
   │
   │   For each step:
   │     1. Call LLM with: system_prompt + tool_declarations + message_history
   │     2. LLM responds with either: tool function call(s) OR final text
   │     3. If tool calls:
   │           - emit("tool_call_start", ...)
   │           - if requires_approval: persist pending state, emit
   │             ("tool_call_pending_approval"), STREAM ENDS (resume via /decide/)
   │           - else: dispatch to REGISTRY[name].run(args, ctx) inline
   │           - emit("tool_call_result", summary) on success
   │           - emit("tool_call_error", ...) on ToolError
   │           - emit("sources", deduped_chip_dicts)  ← per-tool source emission
   │           - feed result back to LLM as a function-response message
   │     4. If text: stream chars via emit("answer_delta", ...)
   │
   │   When loop exits (final text done OR step cap hit):
   │     - (Phase 4.2) _rank_sources_by_citation reorders the final source list
   │     - emit("sources", reordered)
   │     - emit("done", session_id)
   │     - update AgentRun.status, AgentRun.final_answer_text
   │
   ▼
[Browser SSE consumer in agentApi.runNdjsonStream]
   │
   ▼
[useSpotlight buildStreamHandlers]
   ├── onDelta:        ask.answer += text
   ├── onToolStart:    ask.toolEvents.append({status: "pending", ...})
   ├── onToolResult:   ask.toolEvents[match].status = "done", + summary
   ├── onSources:      ask.answerSources = sources  (replaces wholesale)
   ├── onDone:         isStreaming=false, promote to turns[]
   ├── onPendingApproval: paused state, render ApprovalCard
   └── onError:        ask.askError, promote
   │
   ▼
[SpotlightOverlay TurnView]
   ├── ToolProgressList renders pending/done tool chips with spinners
   ├── ReactMarkdown renders the streamed answer with citation rewriter
   │   (rewriteCitations converts [task:18] → [*Title*](spotlight-citation:task:18)
   │    so the markdown <a> override (CitationLink) can open UrlLinkModal preview)
   └── Source chip row below the answer (click → navigate to entity)
```

---

## 4. Frontend layer (what runs in the browser)

All paths below are under `genos-frontend/src/features/spotlight/`.

### Components

- **[SpotlightOverlay.tsx](../genos-frontend/src/features/spotlight/SpotlightOverlay.tsx)** — the React tree. Renders the sheet, the input row (with SearchIcon + input + usage pill + Ask + History icon), the `ConversationPanel` (renders live turns OR history view), and the search-results scroll list. Layout flips via CSS `order`: in agent mode, conversation panel takes the top of the sheet, input row sits at the bottom (chat-style).

- **`ConversationPanel`** (inside the same file) — scrollable region above the input in agent mode. Three render modes driven by `historyMode`:
  - `"closed"` → live turns + in-flight ask
  - `"list"`   → recent past sessions
  - `"detail"` → one past session's Q&A (read-only archive)

- **`TurnView` / `HistoryArchiveTurn`** — one Q&A unit. Both share `_markdownAnswerSx(isDark)` for typography and the `CitationLink` markdown anchor override so live and archived answers render identically.

- **[SpotlightResultItem.tsx](../genos-frontend/src/features/spotlight/SpotlightResultItem.tsx)** — one row in the typeahead results. A `spotlight_answer` row (a collected past answer) gets a distinct icon/label.

- **Previous-answer reuse** — a `spotlight_answer` result has no entity page to navigate to; clicking it calls `useSpotlight.showStoredAnswer`, which injects the stored `answer_text` + `answer_sources` into `ask` so the existing `TurnView` renders the past Q&A — with clickable inline citations and source chips — without a network call. See §7 *Answer reuse*.

### State (the hook)

**[useSpotlight.ts](../genos-frontend/src/features/spotlight/useSpotlight.ts)** owns:

```
ask        — in-flight turn: { isStreaming, answer, answerSources, toolEvents,
                                pendingApproval, sessionId, turnId, askedQuery, askError }
turns      — completed turns this session (CompletedTurn[], capped, persisted to
             localStorage with 4h TTL)
results    — typeahead results
query      — search input (debounced for fetch, deferred for highlight)
isOpen     — overlay visibility
dailyUsage — { used, limit, is_unlimited } from /api/v2/agent/usage/

historyMode      — "closed" | "list" | "detail"
historySessions  — AgentSessionSummary[] (top 20, loaded on openHistory)
historyDetail    — AgentSessionDetail | null (loaded when a row clicked)
```

Key behaviors:
- **Cmd-K toggle** — global keydown listener (`useEffect` at module mount). Esc closes (with modal-precedence guard so layered modals consume Esc first).
- **Stream gating** — each ask gets a fresh monotonic `turnId`; stream handlers gate `setAsk` on `prev.turnId === askedTurnId` so a late event from a superseded turn can't corrupt the current one.
- **Idempotent promotion** — `promoteCurrentTurn(turnId)` snapshots `ask` into `turns`; deduped via `promotedTurnIdsRef` so both `onDone` and `onError` are safe (whichever arrives second is a no-op).
- **Conversation restore** — on mount, `{sessionId, turns}` is loaded from `localStorage` (4h TTL, versioned) so a page reload doesn't lose the conversation context.

### API client

**[agentApi.ts](../genos-frontend/src/services/agentApi.ts)** — typed wrappers around fetch:

- `askAgentStream(...)`           → POST /agent/ask/, dispatches NDJSON events to handlers
- `decideAgent(...)`              → POST /agent/decide/, also NDJSON (resumes a paused run)
- `fetchAgentUsage(...)`          → GET /agent/usage/
- `fetchAgentSessions(...)`       → GET /agent/sessions/ (history list)
- `fetchAgentSessionDetail(...)`  → GET /agent/sessions/<id>/ (history detail)

The NDJSON parser is a hand-rolled `ReadableStream.getReader()` loop in `runNdjsonStream` — axios isn't used here because it doesn't expose streaming bodies. Each newline-delimited JSON line is dispatched to one of the typed handlers above.

### Citation rendering

The model is instructed (by [prompts.py](../backend_django/origin/search_engine/agent/prompts.py)) to cite entities inline as `[task:123]`, `[chat:dm:9:thread:4]`, `[note:personal:50]`, `[project:5]`.

The frontend `rewriteCitations(answer, sourcesById, ts)` (in SpotlightOverlay.tsx) walks those tokens with a regex and rewrites resolved ones to:

```
[*Entity Title*](spotlight-citation:task:123)
```

Then ReactMarkdown's `components={{ a: CitationLink }}` override picks up the `spotlight-citation:` href and renders a clickable button that calls `onPreview(source)` — opening the existing [UrlLinkModal](../genos-frontend/src/components/modals/UrlLinkModal.tsx) on top of Spotlight (z-index bumped to 13200 so it sits above the overlay backdrop at 13100). Unresolved tokens are left as literal `[task:123]` text.

The same rewriter machinery now powers the History archive view too — sources are reconstructed server-side from `AgentStep.result_json` so archived citations are clickable just like live ones.

---

## 5. The agent loop (the heart of the backend)

All paths below are under `backend_django/origin/search_engine/agent/`.

### Entry point

[controller.py](../backend_django/origin/search_engine/agent/controller.py) exports `run_agent(query, ctx, emit, *, run_id=None, prior_turns=None, prior_summary=None, web_search_enabled=True, trace_hook=None)`.

```python
def run_agent(...):
    messages = _build_initial_messages(query, prior_turns, prior_summary)
    return _drive_loop_with_critique(messages, ctx, emit, run_id, ...)
       # which usually delegates to _drive_loop unless RAG_AGENT_SELF_CRITIQUE
```

### The step loop

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │                       _drive_loop(messages, ctx, emit)              │
   └────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
            ┌────────────────────────────────────────┐
            │ step = client.generate_step(           │
            │           messages,                    │
            │           tools=_build_tool_declarations(ctx),
            │           system=AGENT_SYSTEM_PROMPT)  │
            └────────────────────────────────────────┘
                                    │
                ┌───────────────────┴────────────────────┐
                │                                        │
                ▼                                        ▼
       tool calls present?                       no calls — final text
                │                                        │
                ▼                                        ▼
      for each function_call:                    for each text delta:
        emit("tool_call_start", ...)               emit("answer_delta", text)
              │                                        │
       requires_approval?                              ▼
              │                                  done — exit loop
       ┌──────┴──────┐
       │             │
       ▼             ▼
    YES — pause     NO — run inline:
    - persist        result = REGISTRY[name].run(args, ctx)
      pending state  │
    - emit           ├── ToolError? emit("tool_call_error")
      pending_       │
      approval       └── ok? emit("tool_call_result", summary)
    - STREAM ENDS                  │
      (resume via                  ▼
      /decide/)              build chip sources from result
                             dedupe into seen_sources_by_id
                             emit("sources", chip_list)
                                   │
                                   ▼
                             append function_response to messages
                             continue loop (next step)
```

Loop exits via:
- **Final text** (model emits text instead of a function call) → exit normally, emit `done`.
- **Step cap** (`MAX_STEPS = 8` by default) → exit with `status="step_cap"`.
- **Provider error** → emit `error`, exit.

### Optional wrappers

- **`_drive_loop_with_critique`** ([controller.py](../backend_django/origin/search_engine/agent/controller.py)) — when `RAG_AGENT_SELF_CRITIQUE=true`, buffers ALL events from `_drive_loop`, then runs ONE critique LLM call against the draft answer + captured tool results. Replays events with the answer possibly swapped for a revised version. Default off (latency tax +68%).

- **Phase 4.2 chip ranking** — just before the final `done` event, `_rank_sources_by_citation(answer_text, sources)` re-sorts the source list so cited entities surface leftmost in the chip row.

### Persistence per step

Every step writes an `AgentStep` row:

```python
AgentStep(
    run_id=...,
    step_index=N,
    tool_name="search_knowledge_base" | "list_tasks" | ... | "",
    arguments_json={...},
    summary="…",            # one-line for the UI
    result_json={...},      # full tool output, server-side only
    answer_text="",         # text-only steps store the answer here
)
```

`result_json` is the data the History detail endpoint replays through the source builders to rebuild clickable citations for archived turns (see §7).

### Multi-turn context

When the user has a prior session, [agent_views.py](../backend_django/origin/search_engine/agent_views.py) loads either:
- The last `SESSION_MAX_PRIOR_TURNS=3` verbatim Q&A pairs (default), or
- Up to `_ROLLING_SUMMARY_LOAD_CAP=20` turns when `RAG_SESSION_ROLLING_SUMMARY=true`, routed through [multi_turn.build_prior_context](../backend_django/origin/search_engine/agent/multi_turn.py) which summarises older turns into a single context note via one LLM call.

The verbatim turns are prepended to `messages` as (user, assistant) pairs; the summary (if any) is prepended as an assistant "note to self": `[Context recap from earlier in this conversation: <summary>]`.

---

## 6. Tools (~25, split into read + write)

All tools live in [agent/tools/](../backend_django/origin/search_engine/agent/tools/), each in its own module, each exporting a single `Tool` instance. The `REGISTRY: dict[str, Tool]` in [tools/base.py](../backend_django/origin/search_engine/agent/tools/base.py) aggregates them; the controller dispatches a function call by name.

### Read tools (execute inline)

- **`search_knowledge_base`** — the one open-ended retrieval tool. Wraps `search.py`'s hybrid pipeline and returns top-K matches (chunk + entity + snippet + matched terms).
- **Structured queries**: `list_tasks`, `fetch_task`, `list_projects`, `get_project_summary`, `get_team_members`, `get_current_user`, `fetch_note`, `fetch_chat_thread`.
- **Analytics (Phase 15)**: `get_task_throughput_stats`, `get_top_task_closers`, `get_project_activity_ranking`, `get_workload_distribution`, `get_stale_tasks`.
- **External integrations**: `fetch_pr`, `list_pr_comments`, `list_pr_reviews`, `list_pr_commits`, `list_pr_files` (GitHub via the user's OAuth token); `list_calendars`, `list_calendar_events` (Google Calendar). Both gated by `UserFeatureAccess` + OAuth presence checks.
- **`search_web`** — Tavily (gated by `RAG_WEB_SEARCH_ENABLED` + feature access).

### Write tools (require approval)

Flagged `requires_approval=True` on the Tool instance: `create_task`, `update_task`, `assign_task`, `add_comment`, `create_note`, `update_note`, `create_calendar_event`, `update_calendar_event`, `delete_calendar_event`.

When the LLM emits one of these, the controller:
1. Persists the call as `pending` on AgentRun + mints a one-shot `pending_approval_token`.
2. Emits `tool_call_pending_approval` with `{tool_name, arguments, approval_token, run_id}`.
3. Stops the stream (StreamingHttpResponse ends).

Frontend renders an ApprovalCard. User clicks Approve or Reject → POST `/api/v2/agent/decide/` with `{run_id, approval_token, decision}`. The view validates ownership + token, calls `resume_agent` which rebuilds messages from the persisted AgentRun + steps and re-enters the loop (running or skipping the tool depending on decision).

### Source emission per tool

`_ui_source_for_match` (for `search_knowledge_base` matches) and `_ui_sources_from_tool_result` (for structured tools) in [controller.py](../backend_django/origin/search_engine/agent/controller.py) build chip-shaped dicts (`entity_id`, `entity_type`, `title`, `task_id`/`chat_id`/`note_id`/`project_id` for navigation). These deduplicate into `seen_sources_by_id` and stream to the frontend as `sources` events. `list_tasks` and `get_stale_tasks` also emit one unique project source per project_id so inline `[project:N]` citations resolve to clickable links.

### ACL

`ToolContext(team_id, user_id)` is passed to every `Tool.run`. Each tool performs its own auth check (e.g. `list_tasks` verifies project membership via `ProjectMember`); search retrieval ACL is enforced in the OpenSearch query filter (`team_id` term + visibility fields).

---

## 7. Search engine (the OpenSearch side)

All paths below are under `backend_django/origin/search_engine/`.

### Index shape

One OpenSearch index, chunk-level documents. Schema built dynamically by [index_config.py](../backend_django/origin/search_engine/index_config.py) `build_mappings()` so the `embedding` field's `dimension` always matches the active provider (read at startup).

Each chunk doc includes:
- **Identifiers**: `entity_type` (`chat`|`task`|`note`|`thread_summary`|`note_summary`|`todo`|`conversation`|`spotlight_answer`), `entity_id`, `chunk_id`, `chunk_type`
- **Content**: `title`, `search_text`, `snippet_text`
- **Vector**: `embedding` (dimension set by the active provider; see Embeddings)
- **Metadata**: `updated_at`, `text_hash` (for re-embed skip)
- **ACL**: `team_id` + `acl_user_ids` (the flat list of user UUIDs allowed to read the chunk; the retrieval filter requires the requesting user be in it)
- **Backreferences for navigation**: `task_id`, `chat_id`, `note_id`, `project_id`, `related_entity_ids`, etc.
- **Answer-reuse provenance** (`spotlight_answer` chunks only, stored-only/not analyzed): `answer_text` (full answer w/ citation tokens), `answer_sources` (the cited sources) — see *Answer reuse* below.

### Chunkers ([chunkers/](../backend_django/origin/search_engine/chunkers/))

Primary content families:
- **`chat_chunker.py`** — DM/GM/PM/MDM messages and thread replies. Includes "thread anchor" chunks that bundle the parent + first replies.
- **`task_chunker.py`** — task title + body + comments + linked notes.
- **`note_chunker.py`** — BlockNote-document-aware splitting (Phase 9: heading-aware, sentence-aware).

Plus several derived / auxiliary lanes that follow the same `Chunk` contract:
`thread_summary_chunker.py` and `note_summary_chunker.py` (LLM summaries),
`todo_chunker.py`, the per-user `conversation_chunker.py` (your own past Q&A),
and `spotlight_answer_chunker.py` (the team-shared answer-reuse lane — see *Answer reuse* below).

Each chunker produces `Chunk` dicts that `ingestion.py` then embeds + writes.

### Ingestion ([ingestion.py](../backend_django/origin/search_engine/ingestion.py))

Triggered by:
- **Live signals** on app model save/delete (Django signals).
- **Manual reindex** via `manage.py opensearch_setup --recreate` or `manage.py reindex_*`.

Two efficiencies:
- **Hash-based skip** — `text_hash` field; on re-index, chunks whose hash and `embedding_model` match the existing doc are skipped (no re-embed, no PUT).
- **Batched embedding** — chunks are embedded in batches (`embed_texts`) via the configured provider.

### Embeddings ([embeddings/](../backend_django/origin/search_engine/embeddings/))

Provider-neutral protocol ([base.py](../backend_django/origin/search_engine/embeddings/base.py) `Embedder`). Two implementations:
- **`gemini_embedder.py`** / Vertex (default, 1536-dim)
- **`openai_embedder.py`** (alternative)

Picked at runtime by `SEARCH_ENGINE["EMBEDDING_PROVIDER"]`. The active provider's `model_name` and `dimensions` flow into the index mapping AND every chunk's stored `embedding_model` so the re-embed skip check can detect mismatches.

**Two-tier query embedding cache** ([embeddings/\_\_init\_\_.py](../backend_django/origin/search_engine/embeddings/__init__.py)):
- **L1**: bounded `lru_cache(maxsize=256)` per worker process. Zero-network on within-burst repeats (typeahead backspace).
- **L2**: Redis-backed via Django cache framework, keyed `rag:emb:<model>:<sha256[:24]>`, TTL `RAG_EMBEDDING_CACHE_TTL_S=600`. Survives worker restarts and is shared across pods. 281× speedup measured on a cross-worker cache hit. Cache failures log + fall through to the provider API (never break embedding).

Only QUERY embeddings are L2-cached. Document embeddings during ingestion are batched + one-shot, so caching them is wasted Redis.

### Hybrid retrieval ([search.py](../backend_django/origin/search_engine/search.py))

The pipeline order is documented in §3A above. Key knobs:

| Setting | Default | Effect |
|---|---|---|
| `RAG_USE_QUERY_REWRITE` | `false` | LLM expands query into N variants pre-search |
| `RAG_REWRITE_NUM_VARIANTS` | `3` | How many variants |
| `RAG_REWRITE_ORIGINAL_WEIGHT` | `2.0` | RRF weight of the original query vs rewrites |
| `RAG_USE_RERANKER` | `false` | Run a cross-encoder/LLM reranker on top-K |
| `RAG_RERANKER_PROVIDER` | `"llm"` | `"llm"` \| `"cohere"` \| (stubs) `jina`/`local`/`vertex_ranking` |
| `RAG_RERANK_LOCK_TOP_N` | unset | Protect top-N RRF hits from reshuffle |
| `RAG_FRESHNESS_DECAY_HALF_LIFE_DAYS` | unset | Exponential decay on `updated_at` |
| `RAG_DEDUP_BY_TEXT_HASH` | `true` | Drop near-identical bodies |
| `RAG_MIN_SCORE` | `0.040` | Absolute floor (post-fusion) |
| `RAG_THRESHOLD_MIN_SURVIVORS` | `3` | Keep top N if relative threshold would leave fewer |
| `RAG_REPRESENTATION_FLOOR` | `false` | Force ≥1 per matching entity_type |
| `RAG_BM25_TITLE_BOOST` | `3` | BM25 multiplier on title field |
| `RAG_BM25_SNIPPET_BOOST` | `2` | BM25 multiplier on snippet field |

All knobs are flag-gated and reversible. A/B them via `manage.py agent_eval_compare` — see [SPOTLIGHT_OPTIMIZATION_ROADMAP.md](SPOTLIGHT_OPTIMIZATION_ROADMAP.md) for measured outcomes of each.

### Answer reuse — the `spotlight_answer` lane

Completed AI answers are collected back into the index so a teammate who later asks a similar question finds the past answer in typeahead instead of paying for the agent to re-derive it. It's the **team-shared** sibling of the per-user `conversation` memory lane (`chunkers/conversation_chunker.py`), built by [chunkers/spotlight_answer_chunker.py](../backend_django/origin/search_engine/chunkers/spotlight_answer_chunker.py).

- **What's collected** — each clean `AgentRun` (`status="done"`, non-empty answer) becomes one `entity_type="spotlight_answer"` chunk. `search_text` is `"Q: …\nA: …"` so a recall query matches either side; `title` is the original question. v1 collects **single-turn runs only** (a follow-up turn can lean on a prior turn's private source its own steps don't record).

- **ACL — leak-proof, fail-closed** — the answer *body* can quote private content (not just the chips), so the chunk's `acl_user_ids` is set to the **intersection of every source's ACL** — the answer is visible only to users who could have seen *all* of its evidence. Computed in [agent/source_visibility.py](../backend_django/origin/search_engine/agent/source_visibility.py) `shareable_acl_for_sources`, reusing the `agent/acl.py` membership helpers. Any source we can't classify, a single-person/DM/personal-note source (intersection `< 2`), or a run with no internal source → the answer is **dropped**; a previously-collected answer whose source later turned private is purged on the next full reindex via an empty-`EntityChunks` tombstone.

- **Provenance** — the chunk stores stored-only `answer_text` (full answer with inline `[type:id]` citation tokens) and `answer_sources` (the SpotlightResult-shaped source dicts), plus `related_entity_ids`. Sources are rebuilt from `AgentStep.result_json` via `reconstruct_sources_for_run(run)` in [controller.py](../backend_django/origin/search_engine/agent/controller.py) — the same builder the History detail view uses. The frontend renders these as the clickable "Previous answer" card (§4).

- **Surfacing — typeahead in, agent grounding out** — `_build_filter` in [search.py](../backend_django/origin/search_engine/search.py) keys off the resolved `mode`: the lane is **included** for `mode="typeahead"` (the `/api/v2/search/` path, `for_agent=False`) but **excluded** (alongside the `conversation` lane) for `ai_search`/`eval`, so a past answer surfaces to a searching user yet never feeds back into the agent's own grounding (avoiding an answer→grounding→answer loop). The typeahead request needs no change — it already searches "all".

- **Indexing & ops** — batch-only (zero added latency to the live ask), like the summary/conversation lanes. The two provenance fields are stored-only and **additive**, so a live index takes them with no recreate / re-embed:

  ```
  python manage.py opensearch_setup --update-mapping            # additive put_mapping — new fields only
  python manage.py opensearch_reindex --entity-types spotlight_answer
  ```

- **Known v1 limitation** — the question text (`title`/`search_text`/`snippet`, from `run.query`) is not itself ACL-gated; exposure is still bounded to the source-audience intersection, never the whole team. Gating the question text, multi-turn session-union collection, and dedup by normalized question are future work.

---

## 8. LLM provider abstraction

[llm/\_\_init\_\_.py](../backend_django/origin/search_engine/llm/__init__.py) exposes `get_client()` which returns a `ModelClient` adhering to the protocol in [llm/base.py](../backend_django/origin/search_engine/llm/base.py).

Two implementations:
- **[gemini_client.py](../backend_django/origin/search_engine/llm/gemini_client.py)** — Google Gemini (default, `claude-opus-4-7` style invocation isn't the right comparison — this is Gemini 2.5 Pro for the main agent). Uses native function-calling API. `GEMINI_USE_VERTEX=true` routes through Vertex AI with the same `Embedder` auth.
- **[claude_client.py](../backend_django/origin/search_engine/llm/claude_client.py)** — Anthropic Claude (alternative). Same function-calling protocol mapped to Anthropic's tool_use API.

Picked at runtime by `SEARCH_ENGINE["LLM_PROVIDER"]`. Per-call model overrides (e.g. for the reranker or summary path) are passed via `model_override=...` on `generate_step` so a fast model (Flash, Haiku) can be used for cheaper sub-tasks.

The protocol exposes one method:

```python
def generate_step(messages, system, tools, *, model_override=None) -> StepResult:
    # StepResult has: function_calls (list) | text_chunks (iterator)
```

The controller doesn't know which provider it's talking to — only the protocol.

---

## 9. Persistence (Postgres)

Three Django models in [search_engine/models.py](../backend_django/origin/search_engine/models.py):

```
AgentSession                       AgentRun                    AgentStep
─────────────                       ────────                    ─────────
session_id (UUID, PK)              run_id (UUID, PK)            step_id (auto, PK)
team_id        (indexed)           team_id          (indexed)   run_id (FK → AgentRun)
user_id        (indexed)           user_id          (indexed)   step_index
created_at                         query                        tool_name
last_active_at                     status                       arguments_json
                                   final_answer_text            summary
                                   error_message                result_json
                                   pending_approval_token       answer_text
                                   session_id (FK, SET_NULL)    error
                                   started_at                   created_at
                                   finished_at
```

Used for:
- **Multi-turn memory** — `_load_prior_turns(session, max_turns)` returns the last N (query, final_answer) pairs.
- **Daily usage limit** — `AgentRun.objects.filter(user_id, started_at__gte=today)`.
- **Pause / resume** — `pending_approval_token` is the one-shot UUID required (with `run_id`) on `/decide/`.
- **History list** — `AgentSession.objects.filter(team_id, user_id).order_by("-last_active_at")[:20]`.
- **History detail** — joins `session.runs` (filtered to ones with `final_answer_text` OR `error_message`), plus `run.steps` prefetched, with sources reconstructed via `reconstruct_sources_for_run(run)` (in [agent/controller.py](../backend_django/origin/search_engine/agent/controller.py), shared with the `spotlight_answer` chunker) for clickable archived citations.
- **Eval traces** — when `--judge` is on, every case writes its `tool_results` and judge scores to `agent/evals/runs/<ts>.jsonl`.

---

## 10. Cross-cutting concerns

### ACL

- **At indexing time**: every chunk doc carries `team_id` + entity-type-specific visibility fields (`project_id`, `chat_members`, ...).
- **At retrieval time**: `_build_filter(...)` in [search.py](../backend_django/origin/search_engine/search.py) adds an OpenSearch bool filter: `team_id` term + per-entity-type visibility (e.g. `chat_members` contains user_id for chat-type chunks). No chunk the user can't see ever appears in a result.
- **At tool time**: every `Tool.run` performs its own auth check using `ToolContext(team_id, user_id)`. E.g. `list_tasks` verifies project membership via `ProjectMember.objects.filter(...)`; ACL-denied tasks 404.
- **At inline citation click**: opening UrlLinkModal navigates to the entity route, which has its own ACL check. Citations to entities the user has since lost access to fail at modal/page level, not at the agent layer.

### Caching layers

| Layer | What | TTL | Notes |
|---|---|---|---|
| L1 query embed | `lru_cache(256)` per worker | infinite (LRU eviction) | Zero-network on typeahead backspace |
| L2 query embed | Redis `rag:emb:<model>:<sha[:24]>` | `RAG_EMBEDDING_CACHE_TTL_S=600` | Cross-worker, cross-restart; logs every 25th hit |
| OpenSearch shard | Lucene query cache | OS-managed | Built-in benefit on repeat-prefix searches |
| (potential) Rewrite | Not yet implemented | n/a | Flagged as Phase 5.1-ish follow-up |

### Streaming

- **Transport**: NDJSON over POST (not SSE-EventSource — POST avoids query payloads in access logs).
- **Server side**: `StreamingHttpResponse(_stream_ndjson(...), content_type="application/x-ndjson")`. `nginx_buffering` header disabled so events flush incrementally.
- **Client side**: `ReadableStream.getReader()` loop, line-delimited JSON, dispatch to typed handlers. No third-party SSE library.

### Observability

- **Per-run**: `AgentRun` + `AgentStep` rows persist everything except the final per-step summary text that streams to the client.
- **Per-event**: an optional `trace_hook` parameter on `run_agent` lets eval harnesses capture full tool results out-of-band without polluting the SSE stream. Used by the eval runner's `_capture_trace`.
- **TTFT instrumentation**: `_ts_emit` wrapper in [evals/runner.py](../backend_django/origin/search_engine/agent/evals/runner.py) captures the timestamp of the first non-empty `answer_delta` event, written to `CaseResult.ttft_ms` and `--judge` JSONL.
- **L2 cache visibility**: sparse INFO log every 25th hit.

### Configuration

All knobs live in `SEARCH_ENGINE["..."]` in [settings.py](../backend_django/apis/settings.py). Provider env vars (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `TAVILY_API_KEY`, GitHub/Google OAuth client IDs) are read directly. Switching primary models is documented in [MODEL_SWITCHING.md](MODEL_SWITCHING.md).

---

## 11. Where to dig deeper

The phase docs cover the per-feature implementation details — start with the overview here, then drop into the one you need.

### Implementation phase docs

| Doc | Topic |
|---|---|
| [01_agentic_ai_opensearch_mvp_roadmap.md](01_agentic_ai_opensearch_mvp_roadmap.md) | Original MVP plan |
| [02_opensearch_mvp_search_engine_design.md](02_opensearch_mvp_search_engine_design.md) | Search engine design decisions |
| [021_opensearch_chat_rag_chunking_summary.md](021_opensearch_chat_rag_chunking_summary.md) | Chunking strategy for chat data |
| [03_opensearch_mvp_implementation.md](03_opensearch_mvp_implementation.md) | First implementation pass |
| [04_spotlight_agentic_ai_implementation.md](04_spotlight_agentic_ai_implementation.md) | Spotlight UI shell phases 1-2 |
| [05_agent_phase3_implementation.md](05_agent_phase3_implementation.md) | The tool-calling agent loop (this is the deep dive on §5) |
| [06_agent_phase4_implementation.md](06_agent_phase4_implementation.md) | Hardening, safety, observability |
| [07_agent_phase5_implementation.md](07_agent_phase5_implementation.md) | Provider abstraction + Claude adapter |
| [08_agent_phase6_implementation.md](08_agent_phase6_implementation.md) | RAG quality: retrieval evals, freshness, dedup, reranker |
| [09_agent_phase7_implementation.md](09_agent_phase7_implementation.md) | Write-tool approval protocol (the deep dive on §6 write tools) |
| [10_agent_phase8_implementation.md](10_agent_phase8_implementation.md) | Multi-turn session memory |
| [11_agent_phase9_implementation.md](11_agent_phase9_implementation.md) | Chunking refactor (heading-aware notes + chat context windows) |
| [12_agent_phase10_implementation.md](12_agent_phase10_implementation.md) | LLM query rewriting |
| [13_agent_phase11_implementation.md](13_agent_phase11_implementation.md) | Write-tool surface expansion |
| [14_agent_phase13_14_implementation.md](14_agent_phase13_14_implementation.md) | Internal tool expansion + web search + feature gating |
| [15_url_link_modal_implementation.md](15_url_link_modal_implementation.md) | UrlLinkModal (used by inline-citation preview) |

### Operational / reference

| Doc | Topic |
|---|---|
| [SPOTLIGHT_OPTIMIZATION_ROADMAP.md](SPOTLIGHT_OPTIMIZATION_ROADMAP.md) | Sequenced optimisation plan + measured outcomes per phase |
| [SPOTLIGHT_DEMO_VIDEO_SCRIPT.md](SPOTLIGHT_DEMO_VIDEO_SCRIPT.md) | Demo prompts that exercise each path |
| [appendix_opensearch_hybrid_search_guide.md](appendix_opensearch_hybrid_search_guide.md) | OpenSearch hybrid-search internals (BM25 + dense + RRF) |
| [MODEL_SWITCHING.md](MODEL_SWITCHING.md) | How to swap LLM and embedding providers |
| [OPENSEARCH_COMMANDS.md](OPENSEARCH_COMMANDS.md) | Operator commands (recreate index, reindex, etc.) |
| [BACKEND_SCALING.md](BACKEND_SCALING.md) | Scaling notes for the Django backend |
| [HOW_TO_TEST.md](HOW_TO_TEST.md) | Test suites (Django + frontend) |

---

## 12. Common change patterns (so future-you knows where to start)

**"I want to add a new tool the agent can call"**
1. Create `agent/tools/<name>.py` with a `Tool` instance (see existing tools for the shape).
2. Import + add to `REGISTRY` in [tools/\_\_init\_\_.py](../backend_django/origin/search_engine/agent/tools/__init__.py).
3. If it should produce a citation chip, add a branch to `_ui_sources_from_tool_result` in [controller.py](../backend_django/origin/search_engine/agent/controller.py).
4. If it's a write tool, set `requires_approval=True` on the Tool — the loop handles the rest.
5. Add an eval case in [evals/cases.yaml](../backend_django/origin/search_engine/agent/evals/cases.yaml).

**"I want to tune retrieval quality"**
1. Pick (or add) a case in [evals/retrieval_cases.yaml](../backend_django/origin/search_engine/agent/evals/retrieval_cases.yaml).
2. Run `manage.py agent_eval --retrieval` to get the baseline.
3. Toggle the relevant `RAG_*` setting (see §7 table).
4. Run `manage.py agent_eval_compare --b-overrides '{"RAG_FOO": "true"}'` to diff.
5. If it's a net win, change the default in [settings.py](../backend_django/apis/settings.py).

**"I want to add a new entity type to search"**
1. Add a chunker in [chunkers/](../backend_django/origin/search_engine/chunkers/) producing `Chunk` dicts with `entity_type="<new>"` and ACL fields.
2. Update `_build_filter` in [search.py](../backend_django/origin/search_engine/search.py) to add the per-type visibility clause.
3. Add a backend signal handler (or batch reindex step) that triggers `ingestion.upsert_chunks(...)` on model changes.
4. Extend the `EntityType` union in [frontend/...spotlight/types.ts](../genos-frontend/src/features/spotlight/types.ts) and add chip / icon branches.
5. (Optional) Add a `fetch_<entity>` tool so the agent can introspect them.

**"I want to add a new event type to the SSE stream"**
1. Add an `emit({...})` call in [controller.py](../backend_django/origin/search_engine/agent/controller.py) at the right point.
2. Document the shape at the top of [agent_views.py](../backend_django/origin/search_engine/agent_views.py).
3. Add a new variant to the `AgentEvent` discriminated union in [agentApi.ts](../genos-frontend/src/services/agentApi.ts).
4. Add a handler in `buildStreamHandlers` in [useSpotlight.ts](../genos-frontend/src/features/spotlight/useSpotlight.ts).
5. Render it in [SpotlightOverlay.tsx](../genos-frontend/src/features/spotlight/SpotlightOverlay.tsx).

**"I want to change how citations render"**
1. Backend: edit the citation instruction in [prompts.py](../backend_django/origin/search_engine/agent/prompts.py) rule 5.
2. Frontend: edit `rewriteCitations` / `CitationLink` / the regex `CITATION_PATTERN` in [SpotlightOverlay.tsx](../genos-frontend/src/features/spotlight/SpotlightOverlay.tsx) — keep `_INLINE_CITATION_RE` in controller.py in sync.
3. Update [evals/runner.py](../backend_django/origin/search_engine/agent/evals/runner.py) `_CITATION_RE` so eval assertions use the same shape.
