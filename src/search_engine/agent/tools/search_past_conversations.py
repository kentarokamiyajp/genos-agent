"""`search_past_conversations` tool — vector recall over the user's OWN past
agent conversations (C1 / Q2.3, SPOTLIGHT_QUALITY_ARCHITECTURE.md §4.7).

Cross-session memory: the last-N-turns window only covers the CURRENT session,
so a fact the user established in an earlier, ended conversation ("we decided
to rule out framer-motion last week") is invisible to a fresh ask. This tool
searches the per-user `conversation` lane — completed `AgentRun`s indexed by
the conversation chunker — so the model can recall it.

Thin wrapper over `search(entity_types=["conversation"])`: ACL is enforced by
the lane (each conversation chunk's `acl_user_ids = [asker]`), so a user only
ever recalls their own history. The lane is excluded from ordinary
`search_knowledge_base` results, so this is the ONLY path to it.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings

from origin.search_engine.agent.tools.base import Tool, ToolContext, wrap_workspace_content
from origin.search_engine.search import search

_MAX_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {"matches": [], "__summary__": "No query supplied."}

    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, _MAX_LIMIT))

    result = search(
        query=query,
        team_id=ctx.team_id,
        user_id=ctx.user_id,
        entity_types=["conversation"],  # the private per-user lane
        limit=limit,
        for_agent=True,
        rewrite=bool(settings.SEARCH_ENGINE.get("RAG_USE_QUERY_REWRITE", False)),
    )

    matches: list[dict[str, Any]] = []
    for entity in result.get("results", []):
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
                # The original question this past conversation answered.
                "question": entity.get("title"),
                "snippet": wrap_workspace_content(entity.get("snippet") or ""),
                "chunks": chunks,
                "updated_at": entity.get("updated_at"),
            }
        )

    return {
        "matches": matches,
        "__summary__": f"Searched past conversations — {len(matches)} match(es)",
    }


SEARCH_PAST_CONVERSATIONS = Tool(
    name="search_past_conversations",
    description=(
        "Search the user's OWN earlier agent conversations (across sessions) "
        "for something they discussed or decided before. Call this when the "
        "question refers to a prior conversation — 'what did we decide…', "
        "'like I asked last week', 'the plan you gave me earlier' — and the "
        "answer is not in the current session's recent turns. Returns past "
        "Q&A pairs (the earlier question + answer). Only the user's own "
        "conversations are searched; ACL is automatic."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "What to look for in past conversations.",
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max past conversations to return (1-{_MAX_LIMIT}). Default 5.",
            },
        },
        "required": ["query"],
    },
    run=_run,
)
