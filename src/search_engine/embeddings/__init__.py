"""Embedding provider package — factory + public surface.

Public functions (`embed_texts`, `embed_one`, `hash_text`) preserve the
historical embeddings.py API so existing callers in ingestion.py,
search.py, and agent/evals/runner.py don't change.

Provider is chosen by `SEARCH_ENGINE["EMBEDDING_PROVIDER"]`:
    "openai" (default) → `OpenAIEmbedder`
    "vertex"           → `VertexEmbedder` (reuses GEMINI_USE_VERTEX auth)

Two-tier query embedding cache:

  L1 — `embed_one` keeps a bounded per-worker `lru_cache` so the Spotlight
  typeahead path (one embed per keystroke) hits zero network on
  backspace and within-burst repeats. Cache key is `(model_name, text)`;
  a provider/model swap ages stale entries out naturally.

  L2 — Phase 5.1 adds a Redis-backed cache (via Django's cache framework)
  that survives worker restarts and is shared across workers/pods. Same
  key shape (hashed text + model), TTL from `RAG_EMBEDDING_CACHE_TTL_S`
  (default 600s; set to 0 to disable). L2 hit logs at INFO sporadically
  so operators can see the cache is doing something.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Iterable

from django.conf import settings

from origin.search_engine.embeddings.base import Embedder, TaskType

log = logging.getLogger(__name__)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Return one embedding per input text, preserving order.

    Empty strings are sanitised to a single space so the provider API
    doesn't reject the batch; the caller is expected to filter zero /
    placeholder vectors before indexing.
    """
    texts = list(texts)
    if not texts:
        return []
    sanitized = [t if t and t.strip() else " " for t in texts]
    return _get_embedder().embed(sanitized, task_type="document")


def embed_one(text: str) -> list[float]:
    """Embed a single string. Cached on `(model, text)` so repeated
    or near-repeated query embeddings inside one Django worker hit
    memory instead of the provider API. The Spotlight typeahead path
    goes through here on every keystroke.

    Empty / whitespace-only input is intentionally skipped — the
    sanitisation logic in `embed_texts` produces a placeholder vector
    that we don't want to keep in the cache.
    """
    if not text or not text.strip():
        return embed_texts([text])[0]
    embedder = _get_embedder()
    # Cache stores immutable tuples; return a fresh list so callers
    # can't accidentally mutate the cached entry.
    return list(_embed_one_cached(embedder.model_name, text))


# Bounded LRU. 256 entries is plenty for one user's typing burst
# (each prefix of a 16-char query is at most 16 entries) and small
# enough that 1536-dim float lists won't dominate worker memory:
# 256 entries * 1536 floats * 8 bytes ≈ 3 MB.
@lru_cache(maxsize=256)
def _embed_one_cached(model: str, text: str) -> tuple:
    """L1 cache layer for single-text query embeddings. Keys on
    `(model, text)` so a model swap (provider change, dimension
    change, version bump) doesn't return stale vectors — old entries
    simply age out of the LRU. `task_type` is implicit ("query" for
    every cached call) so it's not in the key.

    On L1 miss: checks the L2 (Redis) cache before hitting the
    provider API. On L2 miss: calls the provider, populates L2, and
    lets the LRU cache the return value as L1.
    """
    cached = _l2_get(model, text)
    if cached is not None:
        return tuple(cached)
    embedder = _get_embedder()
    vec = embedder.embed([text], task_type="query")[0]
    _l2_set(model, text, vec)
    return tuple(vec)


# Sparse hit logging — at INFO every Nth L2 hit so operators can
# eyeball the cache is doing real work without flooding logs on a
# busy typeahead day. Tracked per-process; not thread-safe, but the
# counter is best-effort observability — a missed increment doesn't
# matter.
_L2_HIT_COUNTER = {"n": 0}
_L2_HIT_LOG_EVERY = 25


def _l2_enabled() -> bool:
    """L2 is on when TTL > 0. Operators flip it off via
    `RAG_EMBEDDING_CACHE_TTL_S=0`."""
    return int(settings.SEARCH_ENGINE.get("RAG_EMBEDDING_CACHE_TTL_S", 600)) > 0


def _l2_key(model: str, text: str) -> str:
    """Cache key shape: `rag:emb:<model>:<sha256-prefix>`.

    Hashed text (not raw) keeps keys compact and avoids encoding
    issues with weird Unicode. 24 hex chars = 96 bits = collision
    probability ~0 in practice; full 64-char hash is overkill.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"rag:emb:{model}:{h}"


def _l2_get(model: str, text: str) -> list[float] | None:
    """Fetch a cached vector from Redis. Returns None on miss, error,
    or when L2 is disabled. Never raises — embedding the same text
    via the provider API is always a valid fallback.
    """
    if not _l2_enabled():
        return None
    try:
        from django.core.cache import cache  # noqa: PLC0415 — Django ready by call time

        vec = cache.get(_l2_key(model, text))
    except Exception:  # noqa: BLE001 — cache failures must not break embedding
        log.exception("L2 embedding cache GET failed; falling through to API")
        return None

    if vec is not None:
        _L2_HIT_COUNTER["n"] += 1
        if _L2_HIT_COUNTER["n"] % _L2_HIT_LOG_EVERY == 0:
            log.info(
                "embedding L2 cache hit %d (model=%s, text-hash-prefix=%s)",
                _L2_HIT_COUNTER["n"],
                model,
                _l2_key(model, text).rsplit(":", 1)[-1][:8],
            )
    return vec


def _l2_set(model: str, text: str, vec: list[float]) -> None:
    """Store a vector in Redis with the configured TTL. Silent on
    failure — the LRU already has a fresh copy so the next request
    from this worker still benefits even if Redis is unreachable.
    """
    if not _l2_enabled():
        return
    try:
        from django.core.cache import cache  # noqa: PLC0415

        ttl = int(settings.SEARCH_ENGINE.get("RAG_EMBEDDING_CACHE_TTL_S", 600))
        cache.set(_l2_key(model, text), vec, timeout=ttl)
    except Exception:  # noqa: BLE001 — non-fatal
        log.exception("L2 embedding cache SET failed; entry not persisted")


def hash_text(text: str) -> str:
    """SHA-256 of the input text. Used to skip re-embedding unchanged chunks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_active_embedding_model_name() -> str:
    """Model identifier of the currently-active provider. Read by
    `ingestion.py` to populate `RagChunk.embedding_model`, which drives
    the re-embed mismatch check."""
    return _get_embedder().model_name


def get_active_embedding_dimensions() -> int:
    """Vector dim of the currently-active provider. Read by
    `index_config.build_mappings()` so the OpenSearch `knn_vector`
    mapping always matches what we'll write into it."""
    return _get_embedder().dimensions


def _get_embedder() -> Embedder:
    """Return the configured `Embedder` adapter.

    Lazily imports each adapter so a deploy that only uses one provider
    doesn't pay the import cost (and a missing SDK for an unused
    provider doesn't break the app). Raises `RuntimeError` for an
    unknown value rather than silently falling back, so a typo in the
    env var surfaces immediately.
    """
    provider = (settings.SEARCH_ENGINE.get("EMBEDDING_PROVIDER") or "openai").lower()
    if provider == "openai":
        from origin.search_engine.embeddings.openai_embedder import OpenAIEmbedder  # noqa: PLC0415

        return OpenAIEmbedder()
    if provider == "vertex":
        from origin.search_engine.embeddings.vertex_embedder import VertexEmbedder  # noqa: PLC0415

        return VertexEmbedder()
    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER {provider!r}. "
        "Set SEARCH_ENGINE['EMBEDDING_PROVIDER'] to 'openai' or 'vertex'."
    )


__all__ = [
    "Embedder",
    "TaskType",
    "embed_one",
    "embed_texts",
    "get_active_embedding_dimensions",
    "get_active_embedding_model_name",
    "hash_text",
]
