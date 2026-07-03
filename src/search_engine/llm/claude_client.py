"""Claude (Anthropic) adapter for the `ModelClient` interface.

Translates between provider-neutral types (`AgentMessage`,
`FunctionCall`, `ToolDeclaration`) and Anthropic's Messages API.

Two notable shape differences vs. Gemini:

1. **Roles**: Anthropic has only `user` and `assistant`. Tool calls
   live inside assistant messages as `tool_use` content blocks; tool
   results live inside user messages as `tool_result` content blocks.

2. **Tool-use IDs**: every `tool_use` carries an `id`, and the matching
   `tool_result` must reference it via `tool_use_id`. Our neutral
   `AgentMessage` doesn't carry one (Gemini doesn't need it), so we
   synthesize sequential ids (`call_0`, `call_1`, ...) by walking the
   message history in order: each `assistant.function_call` gets the
   next id, and the very next `tool_response` reuses it. The controller
   always appends function-call + function-response turns adjacent, so
   order-based correlation is reliable.

3. **JSON Schema types**: the existing tool definitions use Gemini's
   UPPERCASE form ("OBJECT", "STRING", ...). Anthropic accepts only
   standard lowercase JSON Schema. We normalize in `_normalize_schema`
   below — the tool definitions stay unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import anthropic
from anthropic.lib.streaming import TextEvent
from django.conf import settings

from origin.search_engine.llm.types import AgentMessage, FunctionCall, ToolDeclaration

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Singleton accessor for the Anthropic SDK client."""
    global _client
    if _client is None:
        api_key = settings.SEARCH_ENGINE.get("CLAUDE_API_KEY") or ""
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=claude but CLAUDE_API_KEY is not set. "
                "Get a key from https://console.anthropic.com/ and set "
                "CLAUDE_API_KEY in the environment."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


class ClaudeClient:
    """`ModelClient` adapter backed by Anthropic's Messages API."""

    def generate_step(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDeclaration],
        system_instruction: str,
        *,
        model_override: str | None = None,
    ) -> Iterator[tuple[str | None, FunctionCall | None]]:
        """Stream one model turn against the given history.

        Same contract as `GeminiClient.generate_step` — yields
        `(text_chunk, None)` for incremental text and
        `(None, FunctionCall)` for each function call.
        """
        sdk_messages = _messages_to_anthropic(messages)
        sdk_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": _normalize_schema(t.parameters_schema),
            }
            for t in tools
        ]

        model = model_override or settings.SEARCH_ENGINE["CLAUDE_MODEL"]
        max_tokens = int(settings.SEARCH_ENGINE.get("CLAUDE_MAX_TOKENS", 4096))

        # Opus 4.7 uses extended thinking and the Anthropic API rejects
        # `temperature` on it (400 invalid_request_error). Skip the
        # parameter for that family; the rest of the catalog still
        # honors the 0.2 deterministic-leaning bias we want.
        stream_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_instruction,
            "tools": sdk_tools or anthropic.NOT_GIVEN,
            "messages": sdk_messages,
        }
        if not model.startswith("claude-opus-4-7"):
            stream_kwargs["temperature"] = 0.2

        try:
            with _get_client().messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    # Text deltas: one TextEvent per incremental token
                    # batch. (Helper events from `messages.stream` —
                    # not the raw protocol events.)
                    if isinstance(event, TextEvent):
                        if event.text:
                            yield (event.text, None)
                        continue
                    # End-of-content-block fires as either
                    # RawContentBlockStopEvent (text blocks) or
                    # ParsedContentBlockStopEvent (tool_use blocks,
                    # which carry the fully-assembled `content_block`).
                    # The SDK class hierarchy is awkward, so we
                    # discriminate by event.type + presence of
                    # `content_block` — works across SDK versions and
                    # both event flavors.
                    if getattr(event, "type", None) == "content_block_stop":
                        block = getattr(event, "content_block", None)
                        if block is not None and getattr(block, "type", None) == "tool_use":
                            yield (
                                None,
                                FunctionCall(
                                    name=getattr(block, "name", "") or "",
                                    args=dict(getattr(block, "input", {}) or {}),
                                ),
                            )
                        continue
                    # All other events (message_start, message_delta,
                    # input_json_delta, content_block_start, etc.) are
                    # informational — the high-level helpers above give
                    # us all we need.
        except Exception:
            log.exception("Claude generate_step failed")
            raise


# --------------------------------------------------------------------------- #
# Translation helpers                                                         #
# --------------------------------------------------------------------------- #


def _messages_to_anthropic(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Translate neutral `AgentMessage`s into Anthropic Messages-API turns.

    Synthesizes `tool_use` ids (`call_0`, `call_1`, ...) by walking
    function-call assistant turns in order and reusing the same id
    for the immediately-following `tool_response`. See module docstring.
    """
    out: list[dict[str, Any]] = []
    next_call_index = 0
    pending_tool_use_id: str | None = None  # set after a function_call assistant turn

    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text or ""})
            pending_tool_use_id = None
            continue

        if m.role == "assistant":
            if m.function_call is not None:
                tool_use_id = f"call_{next_call_index}"
                next_call_index += 1
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": m.function_call.name,
                                "input": dict(m.function_call.args),
                            }
                        ],
                    }
                )
                pending_tool_use_id = tool_use_id
                continue
            # Plain text assistant turn.
            out.append({"role": "assistant", "content": m.text or ""})
            pending_tool_use_id = None
            continue

        if m.role == "tool_response":
            if pending_tool_use_id is None:
                # Defensive: this shouldn't happen given the controller
                # always pairs function-call + tool-response. Fall back
                # to a fresh synthetic id so the SDK doesn't reject the
                # request outright; log so the misuse surfaces.
                log.warning(
                    "tool_response without a preceding assistant function_call; "
                    "synthesizing fresh id"
                )
                pending_tool_use_id = f"call_{next_call_index}"
                next_call_index += 1
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": pending_tool_use_id,
                            # Anthropic accepts a string here. JSON-encode
                            # the dict so the model sees structured data.
                            "content": json.dumps(m.function_response or {}, ensure_ascii=False),
                        }
                    ],
                }
            )
            pending_tool_use_id = None
            continue

        raise ValueError(f"Unknown AgentMessage role: {m.role!r}")

    return out


# JSON Schema type names → lowercase. Gemini's tool schemas use the
# uppercase protobuf-derived form ("OBJECT", "STRING", "ARRAY",
# "INTEGER", ...). Anthropic only accepts standard JSON Schema, which
# is lowercase. We rewrite the schema recursively so tool definitions
# stay unchanged.
_TYPE_MAP = {
    "OBJECT": "object",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "NULL": "null",
}


def _normalize_schema(schema: Any) -> Any:
    """Recursively lowercase any uppercase JSON-Schema `type` values."""
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str) and v in _TYPE_MAP:
                out[k] = _TYPE_MAP[v]
            else:
                out[k] = _normalize_schema(v)
        return out
    if isinstance(schema, list):
        return [_normalize_schema(x) for x in schema]
    return schema
