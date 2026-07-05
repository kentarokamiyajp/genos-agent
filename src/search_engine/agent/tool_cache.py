"""C3 — session-scoped tool-result cache (SPOTLIGHT_FUTURE_ARCHITECTURE.md §4).

Within one conversation, a follow-up like "and who's the assignee?"
re-calls the exact tool the previous turn just ran (`list_tasks` with
identical args) — a wasted tool round-trip plus, in the loop, wasted
model context churn. This module is a read-through cache over READ-ONLY
tool results keyed by `(session_id, tool_name, canonical args)`:
`controller._drive_loop` consults it before `tool.run` and stores
successful results after.

Design constraints (mirrors the embeddings L2 cache in
`embeddings/__init__.py` — same never-raise + `django.core.cache`
discipline, so a missing/broken Redis degrades to plain re-execution):

* **Read-only tools only.** Write tools (`requires_approval=True`) are
  never cached, and a successful APPROVED write invalidates the whole
  session's cache — the workspace just changed under it.
* **Invalidation without SCAN.** Django's cache API has no
  delete-by-pattern, so per-session invalidation is a GENERATION
  counter: every key embeds the session's current generation and
  `invalidate_session` just bumps it — O(1), and orphaned entries age
  out via TTL.
* **Freshness tradeoff (documented, deliberate).** Same-user writes
  through the agent invalidate; a TTL (default 300 s) bounds staleness
  from everything else. But ANOTHER user's edit within the TTL can
  still serve a stale read with no signal — acceptable for a team
  workspace (the data is at most TTL-old, like any list-view cache),
  and the reason the feature ships default-OFF pending dogfood.
* **Values are JSON round-tripped** on store so the cached copy is
  detached from the live dict the loop keeps mutating (summary pop,
  source hydration).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from django.conf import settings

log = logging.getLogger(__name__)

_GEN_KEY = "rag:toolcache:gen:{session_id}"


def enabled() -> bool:
    return bool(settings.SEARCH_ENGINE.get("RAG_SESSION_TOOL_CACHE", False))


def _ttl() -> int:
    return int(settings.SEARCH_ENGINE.get("RAG_SESSION_TOOL_CACHE_TTL_S", 300))


def _generation(cache: Any, session_id: str) -> int:
    gen = cache.get(_GEN_KEY.format(session_id=session_id))
    return int(gen) if gen is not None else 0


def _key(session_id: str, gen: int, tool_name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"rag:toolcache:{session_id}:{gen}:{tool_name}:{h}"


def get_cached(session_id: str | None, tool_name: str, args: dict[str, Any]) -> dict | None:
    """Return `{"summary": str, "result": dict}` on hit, else None.
    Never raises — re-running the tool is always a valid fallback."""
    if not session_id or not enabled():
        return None
    try:
        from django.core.cache import cache  # noqa: PLC0415 — Django ready by call time

        entry = cache.get(_key(session_id, _generation(cache, session_id), tool_name, args))
    except Exception:  # noqa: BLE001 — cache failures must not break the loop
        log.exception("session tool cache GET failed; running the tool")
        return None
    if entry is not None:
        log.info("session tool cache HIT: session=%s tool=%s", session_id, tool_name)
    return entry


def store(
    session_id: str | None,
    tool_name: str,
    args: dict[str, Any],
    summary: str,
    result: dict[str, Any],
) -> None:
    """Cache one successful read-only result. Silent on failure."""
    if not session_id or not enabled():
        return
    try:
        # JSON round-trip: detach from the live dict AND guarantee the
        # entry is exactly what `result_json` persistence would hold.
        detached = json.loads(json.dumps(result, default=str))
        from django.core.cache import cache  # noqa: PLC0415

        cache.set(
            _key(session_id, _generation(cache, session_id), tool_name, args),
            {"summary": summary, "result": detached},
            timeout=_ttl(),
        )
    except Exception:  # noqa: BLE001 — non-fatal
        log.exception("session tool cache SET failed; entry not persisted")


def invalidate_session(session_id: str | None) -> None:
    """Bump the session's generation so every existing entry is
    unreachable (orphans age out via TTL). Called after a successful
    APPROVED write tool — the workspace changed, so cached reads from
    before the write must not survive it."""
    if not session_id:
        return
    try:
        from django.core.cache import cache  # noqa: PLC0415

        key = _GEN_KEY.format(session_id=session_id)
        gen = _generation(cache, session_id)
        # Generation key gets a long horizon (not the entry TTL): it only
        # needs to outlive the entries it invalidates.
        cache.set(key, gen + 1, timeout=max(_ttl() * 10, 3600))
        log.info("session tool cache invalidated: session=%s gen=%d", session_id, gen + 1)
    except Exception:  # noqa: BLE001 — non-fatal
        log.exception("session tool cache invalidation failed")
