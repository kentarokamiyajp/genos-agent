"""LLM provider package — factory + neutral types.

`get_model_client()` is the only thing other modules should import
from here. It picks the active adapter based on:

  1. The request-scoped `LlmChoice` set via
     `origin.search_engine.llm.choice.set_llm_choice()` — when present,
     this is the user's saved provider/model.
  2. `SEARCH_ENGINE["LLM_PROVIDER"]` (env-driven server default) when
     no choice is set.

Importing adapters lazily inside each branch means a deploy that only
uses one provider doesn't pay the import cost of the others (and a
missing SDK for an unused provider doesn't break the app).

When a `LlmChoice` is in scope, the returned client wraps
`generate_step` so per-call callers (the agent controller, reranker,
etc.) don't need to know about the user's choice — the model id is
injected automatically. Callers that pass an explicit `model_override`
still win, so existing call sites like the reranker keep working.
"""

from __future__ import annotations

from typing import Iterator

from django.conf import settings

from origin.search_engine.llm.base import ModelClient
from origin.search_engine.llm.choice import LlmChoice, get_llm_choice
from origin.search_engine.llm.types import (
    AgentMessage,
    FunctionCall,
    ToolDeclaration,
)


def _build_adapter(provider: str) -> ModelClient:
    """Return a fresh adapter instance for `provider` ('gemini'|'claude')."""
    if provider == "gemini":
        from origin.search_engine.llm.gemini_client import GeminiClient  # noqa: PLC0415

        return GeminiClient()
    if provider == "claude":
        from origin.search_engine.llm.claude_client import ClaudeClient  # noqa: PLC0415

        return ClaudeClient()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER {provider!r}. "
        "Set SEARCH_ENGINE['LLM_PROVIDER'] to 'gemini' or 'claude'."
    )


class _ChoiceWrappedClient:
    """Wraps a `ModelClient` to inject the user-chosen model id.

    The wrapped `generate_step` passes `model_override=choice.model`
    unless the caller already supplied their own override (which still
    wins — e.g. the reranker pinning a small/fast model).
    """

    def __init__(self, inner: ModelClient, choice: LlmChoice) -> None:
        self._inner = inner
        self._choice = choice

    def generate_step(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDeclaration],
        system_instruction: str,
        *,
        model_override: str | None = None,
    ) -> Iterator[tuple[str | None, FunctionCall | None]]:
        effective_override = model_override or self._choice.model or None
        return self._inner.generate_step(
            messages,
            tools,
            system_instruction,
            model_override=effective_override,
        )


def get_model_client() -> ModelClient:
    """Return the `ModelClient` adapter for the current request.

    Resolution order:
      1. Request-scoped `LlmChoice` (set by `set_llm_choice()`).
      2. `SEARCH_ENGINE["LLM_PROVIDER"]` env-driven default.

    Raises `RuntimeError` for an unknown provider value rather than
    silently falling back, so a typo in the env var surfaces
    immediately.
    """
    choice = get_llm_choice()
    if choice is not None:
        return _ChoiceWrappedClient(_build_adapter(choice.provider), choice)

    provider = (settings.SEARCH_ENGINE.get("LLM_PROVIDER") or "gemini").lower()
    return _build_adapter(provider)


__all__ = [
    "AgentMessage",
    "FunctionCall",
    "ModelClient",
    "ToolDeclaration",
    "get_model_client",
]
