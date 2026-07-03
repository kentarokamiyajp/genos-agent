"""`list_my_mentions` tool — stubbed during the legacy chat retirement.

The legacy `MentionFact` table was dropped in Phase 3. A v3 replacement
will surface from `MessageMention` once the OpenSearch indexer is
rewritten to source from `Channel`/`Message`. Until then, the tool
returns an empty list so the agent's tool catalog stays stable.
"""

from typing import Any

from origin.search_engine.agent.tools.base import Tool, ToolContext


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {
        "since": None,
        "mentions": [],
        "__summary__": (
            "list_my_mentions is temporarily disabled while the chat "
            "indexer migrates to the unified Message schema. Re-enabled "
            "once the OpenSearch chunker reads from v3 MessageMention."
        ),
    }


LIST_MY_MENTIONS = Tool(
    name="list_my_mentions",
    description=(
        "Recent @mentions of the current user. Disabled while the chat "
        "indexer migrates to the unified Message schema; returns an empty "
        "list."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "since": {"type": "STRING", "description": "Unused while disabled."},
            "limit": {"type": "INTEGER", "description": "Unused while disabled."},
        },
    },
    run=_run,
    requires_approval=False,
)
