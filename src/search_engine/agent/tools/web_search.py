"""`search_web` tool — live web search via Tavily.

Extends the agent with public knowledge. The internal `search_knowledge_base`
covers the team's private chats, tasks, and notes; `search_web` covers
everything else — documentation, best practices, external references.

Design:
  * Read-only (requires_approval=False). Web queries leak no private data
    and write nothing, so they run inline like the other read tools.
  * Token budget: each result's `content` is trimmed to _MAX_CONTENT_CHARS.
    Tavily already returns extracted clean text (no raw HTML), so the LLM
    gets dense, readable snippets without further post-processing.
  * search_depth="advanced" — Tavily's deeper crawl; costs 2 credits per
    call vs 1 for "basic", but produces materially better coverage for
    technical questions. Trade-off is acceptable at the default limit of 5
    results per call.
  * Graceful no-op: if TAVILY_API_KEY is missing the tool raises ToolError
    (surfaces in the UI as a tool row "✗") rather than crashing the whole
    agent loop. The model can then tell the user web search is unavailable.

No ACL required — web search results are public by definition.
"""

from __future__ import annotations

from typing import Any

from django.conf import settings

from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.quota import (
    WEB_SEARCH_KEY,
    check_remaining,
    increment_usage,
)

_MAX_CONTENT_CHARS = 600
_MAX_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolError("`query` is required.")

    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, _MAX_LIMIT))

    # --- Per-tier daily quota. ---
    # Free/Pro/Max each get a different `web_search_daily` cap from
    # SEARCH_ENGINE["TIER_QUOTAS"]. Pre-flight check; increment after
    # a successful Tavily response so failed calls don't burn quota.
    allowed, used, web_limit = check_remaining(ctx.user_id, WEB_SEARCH_KEY)
    if not allowed:
        raise ToolError(
            f"You've used all {web_limit} web searches for today. "
            "Upgrade your plan to keep going."
        )

    api_key = (settings.SEARCH_ENGINE.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        raise ToolError(
            "Web search is not configured. "
            "Set TAVILY_API_KEY in environment variables to enable it."
        )

    # Import inside the function so the module can be imported (and the
    # tool registered) even if tavily-python is somehow absent — the error
    # surfaces only when the tool is actually called, not at startup.
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError:
        raise ToolError(
            "tavily-python is not installed. "
            "Run `pip install tavily-python` and restart the server."
        )

    client = TavilyClient(api_key=api_key)
    try:
        resp = client.search(query, max_results=limit, search_depth="advanced")
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Web search failed: {e}")

    # Successful Tavily call → charge one unit against the user's daily
    # web search quota. The increment is best-effort (logs + swallows
    # exceptions inside increment_usage), so a counter outage never
    # blocks the agent from returning results.
    increment_usage(ctx.user_id, WEB_SEARCH_KEY)

    items = [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:_MAX_CONTENT_CHARS],
        }
        for r in (resp.get("results") or [])
    ]

    return {
        "results": items,
        "__summary__": f"Web search '{query[:60]}' → {len(items)} result(s)",
    }


SEARCH_WEB = Tool(
    name="search_web",
    description=(
        "Search the live web for up-to-date information, documentation, "
        "best practices, or any knowledge not found in the team's internal "
        "chats, tasks, and notes. "
        "Use this when the user asks 'how do I…', 'what is the best way to…', "
        "or references external tools, libraries, or concepts the internal "
        "knowledge base is unlikely to cover. "
        "Combine with search_knowledge_base for questions that need both "
        "internal context (e.g. the specific task details) and external "
        "guidance (e.g. how to solve the underlying problem). "
        "Returns titles, URLs, and clean content snippets — no raw HTML."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": (
                    "Concise, specific search query. Prefer technical terms "
                    "over vague phrases — e.g. 'Django zero-downtime migration "
                    "large table' rather than 'how to migrate'."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max results to return (1–{_MAX_LIMIT}). Default 5.",
            },
        },
        "required": ["query"],
    },
    run=_run,
    requires_approval=False,
)
