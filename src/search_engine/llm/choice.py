"""Per-request LLM choice resolution.

The factory in `origin.search_engine.llm.__init__` consults the
`ContextVar` set by `set_llm_choice()` to decide which provider +
model adapter to return for the current request. When unset (no
choice resolved yet, or a non-request code path), the factory falls
back to `settings.SEARCH_ENGINE["LLM_PROVIDER"]` / `GEMINI_MODEL` /
`CLAUDE_MODEL`.

Threading note: `AgentAskView` runs the controller loop on a
`threading.Thread` (see `_stream_ndjson` in `agent_views.py`). Bare
threads do NOT inherit `ContextVar` values from their parent thread,
so callers must either `set_llm_choice()` from inside the worker
thread, or wrap the worker with `contextvars.copy_context().run(...)`.

Singleton note: the Gemini/Claude SDK clients are module-level
singletons keyed on server-wide credentials (API key / service
account). Only the model id is per-user; that flows through each
`generate_step(..., model_override=...)` call. A future change to
per-user *auth* (e.g. customer-supplied API keys) would also need
to unwind the singletons in `gemini_client.py` and `claude_client.py`.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass

from django.conf import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmChoice:
    """Provider + model id pair, normalized lowercase."""

    provider: str  # 'gemini' | 'claude'
    model: str  # e.g. 'gemini-2.5-pro' / 'claude-sonnet-4-6'


_current_choice: ContextVar[LlmChoice | None] = ContextVar("llm_choice", default=None)


def set_llm_choice(choice: LlmChoice | None):
    """Bind `choice` to the current context; returns the reset token.

    Always pair with `reset_llm_choice(token)` in a `finally:` block —
    sync Django reuses worker threads across requests, so a missing
    reset would leak this user's choice to the next request on the
    same worker.
    """
    return _current_choice.set(choice)


def reset_llm_choice(token) -> None:
    _current_choice.reset(token)


def get_llm_choice() -> LlmChoice | None:
    return _current_choice.get()


def _catalog_has(provider: str, model: str) -> bool:
    """True iff `(provider, model)` is in `SEARCH_ENGINE['MODEL_CATALOG']`."""
    catalog = settings.SEARCH_ENGINE.get("MODEL_CATALOG") or []
    return any(e.get("provider") == provider and e.get("model") == model for e in catalog)


def _server_default_choice() -> LlmChoice:
    """The choice implied by env vars when no user preference applies."""
    cfg = settings.SEARCH_ENGINE
    provider = (cfg.get("LLM_PROVIDER") or "gemini").lower()
    if provider == "claude":
        return LlmChoice(provider="claude", model=cfg.get("CLAUDE_MODEL") or "")
    return LlmChoice(provider="gemini", model=cfg.get("GEMINI_MODEL") or "")


def resolve_user_choice(
    preferred_provider: str | None,
    preferred_model: str | None,
) -> LlmChoice:
    """Pick the effective `LlmChoice` for a user.

    - Both fields blank → server default.
    - Provider unknown OR (provider, model) not in `MODEL_CATALOG` →
      server default + warning. This is the stale-preference path: an
      admin removed a model the user had saved. Falling back here keeps
      the request working instead of bubbling an SDK error.
    """
    provider = (preferred_provider or "").lower().strip()
    model = (preferred_model or "").strip()

    if not provider and not model:
        return _server_default_choice()

    if provider not in ("gemini", "claude"):
        log.warning(
            "User has unknown preferred_llm_provider=%r; falling back to server default",
            preferred_provider,
        )
        return _server_default_choice()

    if not model:
        # Provider chosen but no specific model — use that provider's
        # server-configured default model.
        cfg = settings.SEARCH_ENGINE
        if provider == "claude":
            return LlmChoice(provider="claude", model=cfg.get("CLAUDE_MODEL") or "")
        return LlmChoice(provider="gemini", model=cfg.get("GEMINI_MODEL") or "")

    if not _catalog_has(provider, model):
        log.warning(
            "User preference (%s, %s) not in MODEL_CATALOG; falling back to server default",
            provider,
            model,
        )
        return _server_default_choice()

    return LlmChoice(provider=provider, model=model)
