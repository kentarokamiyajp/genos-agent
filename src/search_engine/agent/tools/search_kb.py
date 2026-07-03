"""`search_knowledge_base` tool — agent's entry point into the index.

Thin wrapper over `origin.search_engine.search.search(...)` with
`for_agent=True` so the LLM sees full chunk text (not just snippets)
and gets up to a few chunks per matched entity for grounding.

The controller takes special action for this tool: after a successful
call it promotes the `matches` to citation chips via a `sources`
NDJSON event (deduplicated against earlier searches in the same run).
"""

from __future__ import annotations

from typing import Any

from django.conf import settings

from origin.search_engine.agent.tools.base import Tool, ToolContext, wrap_workspace_content
from origin.search_engine.search import search

_MAX_LIMIT = 20


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {
            "matches": [],
            "__summary__": "No query supplied.",
        }

    entity_types = args.get("entity_types")
    if entity_types is not None and not isinstance(entity_types, list):
        entity_types = None

    try:
        limit = int(args.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, _MAX_LIMIT))

    # Phase 10 — query rewriting is scoped to the agent path only.
    # The Spotlight typeahead never passes through here, so the
    # `RAG_USE_QUERY_REWRITE` flag can't fire an LLM call per keystroke.
    # Enabled by default (see apis/settings.py) — measured +2 net pass /
    # 0 regressions on the agent eval suite (roadmap §1.1).
    use_rewrite = bool(settings.SEARCH_ENGINE.get("RAG_USE_QUERY_REWRITE", False))

    result = search(
        query=query,
        team_id=ctx.team_id,
        user_id=ctx.user_id,
        entity_types=entity_types,
        limit=limit,
        for_agent=True,
        rewrite=use_rewrite,
    )

    # Trim the per-entity payload to what the LLM actually needs:
    # title + snippet + a few chunks. The full UI-shape (scores, ranks,
    # related_entity_ids, etc.) bloats the prompt for no benefit.
    matches: list[dict[str, Any]] = []
    for entity in result.get("results", []):
        # Wrap each chunk's body + the entity snippet in the
        # <workspace_content> boundary marker so the model is steered
        # to treat them as data, not instructions. See
        # `wrap_workspace_content` for the rationale.
        chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "text": wrap_workspace_content(c.get("text", "")),
            }
            for c in (entity.get("chunks") or [])
        ]
        matches.append(
            {
                "entity_type": entity.get("entity_type"),
                "entity_id": entity.get("entity_id"),
                "title": entity.get("title"),
                "snippet": wrap_workspace_content(entity.get("snippet") or ""),
                "chunks": chunks,
                # Surface ids the model might want to pass to fetch_*:
                "chat_type": entity.get("chat_type"),
                "chat_id": entity.get("chat_id"),
                "thread_id": entity.get("thread_id"),
                "task_id": entity.get("task_id"),
                "note_id": entity.get("note_id"),
                "note_type": entity.get("note_type"),
                "project_id": entity.get("project_id"),
                # Surfaced for the source-chip → workspace deep-link.
                # `message_id` lets a chat citation focus the exact
                # matched bubble; `related_entity_ids` is the fallback
                # the frontend reads when chunks pre-date direct
                # task_id / chat_* fields on note rows; the rest match
                # the SpotlightResult shape used by search-result rows
                # so the agent-source path can reuse the same router.
                "message_id": entity.get("message_id"),
                "matched_chunk_types": list(entity.get("matched_chunk_types") or []),
                "matched_terms": list(entity.get("matched_terms") or []),
                "related_entity_ids": list(entity.get("related_entity_ids") or []),
                "updated_at": entity.get("updated_at"),
            }
        )

    return {
        "matches": matches,
        "__summary__": f"Searched knowledge base — {len(matches)} matches",
    }


SEARCH_KNOWLEDGE_BASE = Tool(
    name="search_knowledge_base",
    description=(
        "Hybrid keyword + semantic search over the user's chats, tasks, "
        "notes, and todos. Call this first when the user's question is "
        "vague or you don't know which specific entity to fetch. Returns "
        "the top matches with title, short snippet, and a few full chunks "
        "of text per match. ACL is enforced automatically — the user only "
        "sees results they have access to."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "Natural-language search query.",
            },
            "entity_types": {
                "type": "ARRAY",
                "items": {"type": "STRING", "enum": ["chat", "task", "note", "todo"]},
                "description": (
                    "Optional. Restrict to a subset of entity types. Omit to search all."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max entity-level matches to return (1-{_MAX_LIMIT}). " "Default 10."
                ),
            },
        },
        "required": ["query"],
    },
    run=_run,
)
