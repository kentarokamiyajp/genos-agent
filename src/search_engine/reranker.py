"""Reranker — provider-pluggable second stage over hybrid results.

After hybrid retrieval produces an entity-level result list, an
optional rerank stage can reorder the top-K. Off by default; enabled
per deploy via `SEARCH_ENGINE["RAG_USE_RERANKER"]`.

Providers (`SEARCH_ENGINE["RAG_RERANKER_PROVIDER"]`):

  * `"llm"` (default) — LLM-as-judge via the configured `ModelClient`.
    Quality-vs-speed knobs:
      - `RAG_RERANKER_MODEL`: per-call model override
        (e.g. `gemini-2.5-flash` — ~3× faster than `pro` at equivalent
        rerank quality on our suite).
      - `RAG_RERANK_KEEP_DROPPED`: when True, items the model omits
        are appended in their pre-rerank order instead of being
        discarded. A/B showed this doesn't help on its own — the
        reranker reorders so badly that keeping items just preserves
        the bad order — but kept as a knob for downstream tuning.

  * `"cohere"` — hosted cross-encoder via Cohere v2 Rerank API.
    Requires `COHERE_API_KEY` (gracefully falls back to `"llm"` with
    a warning when unset). Defaults: model `rerank-v3.5`, p95 ~300 ms,
    ~$2 / 1k queries. Cross-encoders trained on (query, passage)
    relevance pairs don't have the LLM judge's paraphrase-wins
    / exact-phrase-loses tradeoff — see session A/B notes in the
    roadmap.

  * (future) `"jina"` — Jina AI Rerank, same shape as Cohere.

  * (future) `"local"` — sentence-transformers cross-encoder
    (`cross-encoder/ms-marco-MiniLM-L-6-v2` or BGE-reranker). Zero
    per-query cost but adds torch/transformers to the image. Implement
    `_rerank_local(...)` behind the same signature.

The eval-suite findings (session 2026-05) show LLM rerank is **net
zero** on this case set: it wins paraphrase queries but loses
exact-phrase queries at similar rate. True cross-encoder models are
trained on (query, passage) → relevance pairs and don't have this
tradeoff; they're the right next step.

Graceful degradation: any error path — malformed JSON, empty
response, model failure, unknown provider — falls back to the
pre-rerank ordering and logs a warning. The reranker should never
crash a query.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from django.conf import settings

from origin.search_engine.llm import AgentMessage, get_model_client

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a search-result reranker. You receive a user query and a
numbered list of workspace items (chats / tasks / notes). Your job is
to return the items in order from MOST RELEVANT to LEAST RELEVANT for
that query.

Rules:
  - Reply with ONLY a JSON array of integers, like [3, 0, 1].
  - Each integer is the index of an item from the input list (0-based).
  - Drop items that are clearly NOT relevant — do not include their
    indices in your output.
  - Do not invent new indices. Do not duplicate indices.
  - If NO items are relevant to the query, reply with [].
  - Do not include any prose, explanation, or markdown — only the
    JSON array.
"""

# Snippet truncation cap when building the rerank prompt. The model
# only needs a short window to judge relevance; longer snippets bloat
# the prompt for no benefit and trigger Anthropic's token limits faster
# than we'd like.
_SNIPPET_TRUNCATE = 200


def _classify_query_type(query: str) -> str:
    """Q2.1 — lightweight, LLM-free query-type signal for per-type fusion
    weighting. Returns "exact" or "paraphrase".

    "exact" = the user is reaching for a specific named thing where the
    keyword/RRF lane is strongest: a quoted phrase, a very short query
    (<= 3 tokens), or a title-cased noun phrase (>= 2 Capitalized words and
    no question word). "paraphrase" = a longer natural-language ask where
    the cross-encoder's semantic lift pays off. Deliberately conservative —
    it only needs to separate the two populations the doc calibrates on, not
    be a general intent classifier.
    """
    q = (query or "").strip()
    if not q:
        return "paraphrase"
    if '"' in q or "'" in q:
        return "exact"
    tokens = q.split()
    if len(tokens) <= 3:
        return "exact"
    qlower = q.lower()
    question_words = ("what", "why", "how", "who", "where", "when", "which", "summar", "explain")
    if not any(qlower.startswith(w) or f" {w}" in qlower for w in question_words):
        capitalized = sum(1 for t in tokens if t[:1].isupper())
        if capitalized >= 2:
            return "exact"
    return "paraphrase"


def _fusion_weight(query: str | None = None) -> float | None:
    """Reranker's share of the fused score when RAG_RERANK_FUSION is on,
    else None (no fusion → providers reorder as before).

    Q2.1: when a per-query-type weight is configured (non-empty
    RAG_RERANK_FUSION_WEIGHT_EXACT / _PARAPHRASE) and a `query` is provided,
    pick the lane via `_classify_query_type`; otherwise fall back to the
    single global RAG_RERANK_FUSION_WEIGHT (so the mechanism is inert until
    calibrated).
    """
    se = settings.SEARCH_ENGINE
    if not se.get("RAG_RERANK_FUSION", False):
        return None
    default = float(se.get("RAG_RERANK_FUSION_WEIGHT", 0.5))
    if query is None:
        return default
    key = "RAG_RERANK_FUSION_WEIGHT_EXACT" if _classify_query_type(query) == "exact" \
        else "RAG_RERANK_FUSION_WEIGHT_PARAPHRASE"
    raw = se.get(key, "")
    if raw == "" or raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _fuse_by_score(
    candidates: list[dict[str, Any]],
    relevance_by_index: dict[int, float],
    weight: float,
    output_k: int,
) -> list[dict[str, Any]]:
    """D2 score fusion: blend each candidate's hybrid (RRF) score with the
    reranker's relevance, then take the top `output_k`. Pure / no I/O.

    `weight` (the reranker's share, clamped to [0,1]); `(1 - weight)` is
    RRF's. The RRF `score` is unbounded (~0.01–0.05, a sum of
    1/(RRF_K+rank)) so it's min-max normalized to [0,1] within the
    candidate set — otherwise a ~0.9 relevance would swamp it. The
    reranker relevance is kept ABSOLUTE (Cohere's relevance_score is
    already calibrated [0,1]; re-normalizing within the set would destroy
    that; the LLM positional proxy is constructed in [0,1]). We fuse the
    RRF *score*, not raw rank, to preserve confidence magnitude (a
    runaway top hit vs a near-tie).

    A candidate the reranker dropped (absent from `relevance_by_index`)
    contributes relevance 0.0 — so it is NOT removed, it degrades to its
    weighted RRF share `(1-weight)·rrf_norm`. That lets a high-RRF hit the
    reranker ignored still rank, though a strongly-reranked paraphrase can
    still displace it (the intended fusion behavior, not a hard floor).

    Asymmetry to keep in mind: min-max forces the candidate set's
    lowest-RRF item to 0.0, while the lowest relevance is whatever the
    reranker returned — an inherent fusion tradeoff, not a neutral blend.
    """
    n = len(candidates)
    if n == 0:
        return []
    rrf = [float(c.get("score", 0.0) or 0.0) for c in candidates]
    lo, hi = min(rrf), max(rrf)
    span = hi - lo
    # span == 0 (all-equal RRF) → RRF carries no signal; let relevance
    # decide by giving every candidate the same RRF contribution.
    rrf_norm = [((s - lo) / span if span > 0 else 1.0) for s in rrf]
    w = max(0.0, min(1.0, weight))
    fused = [(1.0 - w) * rrf_norm[i] + w * float(relevance_by_index.get(i, 0.0)) for i in range(n)]
    order = sorted(range(n), key=lambda i: fused[i], reverse=True)
    return [candidates[i] for i in order[: max(output_k, 0)]]


def rerank(
    *,
    query: str,
    entities: list[dict[str, Any]],
    input_k: int,
    output_k: int,
) -> list[dict[str, Any]]:
    """Dispatch to the configured reranker provider.

    Provider is `SEARCH_ENGINE["RAG_RERANKER_PROVIDER"]` (default
    `"llm"`). Unknown providers fall through to the LLM path with a
    warning so a typo in the env var doesn't crash retrieval.
    """
    provider = (settings.SEARCH_ENGINE.get("RAG_RERANKER_PROVIDER") or "llm").lower()
    if provider == "llm":
        return _rerank_llm(query=query, entities=entities, input_k=input_k, output_k=output_k)
    if provider == "cohere":
        return _rerank_cohere(query=query, entities=entities, input_k=input_k, output_k=output_k)
    # Stubs for future cross-encoder providers — implement these and
    # the dispatcher routes traffic automatically.
    if provider in {"jina", "local", "vertex_ranking"}:
        log.warning(
            "RAG_RERANKER_PROVIDER=%r is not yet implemented; falling back to llm. "
            "See agent/evals/runs/ for the latest A/B numbers when this lands.",
            provider,
        )
        return _rerank_llm(query=query, entities=entities, input_k=input_k, output_k=output_k)
    log.warning("RAG_RERANKER_PROVIDER=%r is unknown; falling back to llm.", provider)
    return _rerank_llm(query=query, entities=entities, input_k=input_k, output_k=output_k)


def _rerank_llm(
    *,
    query: str,
    entities: list[dict[str, Any]],
    input_k: int,
    output_k: int,
) -> list[dict[str, Any]]:
    """LLM-as-judge reranker.

    Args:
        query:     the user query (the original `search(...)` `query` arg).
        entities:  entity-level results from `search(...)`. Order is the
                   pre-rerank ranking.
        input_k:   how many of the top entities to send to the model.
                   Smaller = cheaper but riskier (relevant items beyond
                   K never get a second look).
        output_k:  cap on how many entities to return after reranking.

    Falls back to `entities[:output_k]` (the pre-rerank order) if:
        * fewer than 2 candidates (nothing to rerank),
        * the model returns malformed / unparseable output,
        * the model raises mid-call.
    """
    if not entities or input_k <= 1 or output_k <= 0:
        return entities[: max(output_k, 0)]

    candidates = entities[:input_k]
    prompt = _build_user_prompt(query, candidates)

    client = get_model_client()
    msgs = [AgentMessage(role="user", text=prompt)]

    # Reranking is a narrow classification task — no tool use, no
    # multi-step reasoning. A smaller / faster model usually matches
    # the bigger one's quality at a fraction of the latency. Wire a
    # per-call override so operators can point `RAG_RERANKER_MODEL` at
    # e.g. `gemini-2.5-flash` for the agent path without changing the
    # main `GEMINI_MODEL` used for answer generation.
    model_override = settings.SEARCH_ENGINE.get("RAG_RERANKER_MODEL") or None

    try:
        chunks: list[str] = []
        for text, fc in client.generate_step(
            messages=msgs,
            tools=[],
            system_instruction=_SYSTEM_PROMPT,
            model_override=model_override,
        ):
            if text:
                chunks.append(text)
            # Function-call output is unexpected (no tools given); ignore.
            if fc is not None:
                log.warning(
                    "Reranker unexpectedly got a function call from the model "
                    "(name=%s) — ignoring",
                    fc.name,
                )
        raw = "".join(chunks).strip()
    except Exception:  # noqa: BLE001 — surface as a fallback, not a crash
        log.exception("Reranker LLM call failed; falling back to pre-rerank order")
        return candidates[:output_k]

    indices = _parse_indices(raw, valid_range=len(candidates))
    if indices is None:
        log.warning(
            "Reranker returned unparseable output (%r); falling back to " "pre-rerank order",
            raw[:200],
        )
        return candidates[:output_k]

    # D2 score fusion: the LLM judge gives only an ORDER (no calibrated
    # score), so derive a positional relevance — top item ~1.0, last ~0;
    # dropped items are absent (→ relevance 0 in the fuse, kept at their
    # RRF share). Then blend with the RRF score instead of replacing it.
    weight = _fusion_weight(query)
    if weight is not None:
        relevance = (
            {idx: 1.0 - pos / len(indices) for pos, idx in enumerate(indices)} if indices else {}
        )
        return _fuse_by_score(candidates, relevance, weight, output_k)

    # Optional "rerank but never drop" mode: the model's order wins,
    # but any input candidate the model omitted is appended in its
    # original (pre-rerank) position so nothing is lost. Useful when
    # the reranker is over-pruning. Off by default — A/B in the
    # retrieval suite showed it doesn't help on its own (the
    # reranker reorders so badly that keeping all items just preserves
    # the bad order). Kept as a knob for downstream tuning.
    if settings.SEARCH_ENGINE.get("RAG_RERANK_KEEP_DROPPED", False):
        chosen = set(indices)
        tail = [j for j in range(len(candidates)) if j not in chosen]
        indices = indices + tail

    reordered = [candidates[i] for i in indices[:output_k]]
    return reordered


# --------------------------------------------------------------------------- #
# Cohere Rerank v2 — purpose-built cross-encoder                              #
# --------------------------------------------------------------------------- #


def _rerank_cohere(
    *,
    query: str,
    entities: list[dict[str, Any]],
    input_k: int,
    output_k: int,
) -> list[dict[str, Any]]:
    """Reorder via Cohere's hosted Rerank v2 API.

    Cross-encoders trained on `(query, passage) → relevance` pairs
    don't have the LLM judge's paraphrase-wins-exact-phrase-loses
    tradeoff. They're also ~30–50× faster than an LLM rerank call
    (p95 ~300 ms) and ~5× cheaper.

    Falls back to LLM rerank when `COHERE_API_KEY` is unset, so an
    operator can flip `RAG_RERANKER_PROVIDER=cohere` ahead of
    receiving the key without taking the surface down.

    Pre-rerank-order fallback (same as the LLM path) on any network
    or parsing error so the reranker never crashes a query.
    """
    if not entities or input_k <= 1 or output_k <= 0:
        return entities[: max(output_k, 0)]

    api_key = settings.SEARCH_ENGINE.get("COHERE_API_KEY") or ""
    if not api_key:
        log.warning(
            "RAG_RERANKER_PROVIDER=cohere but COHERE_API_KEY is unset; "
            "falling back to llm reranker."
        )
        return _rerank_llm(query=query, entities=entities, input_k=input_k, output_k=output_k)

    candidates = entities[:input_k]
    documents = [_cohere_doc_text(c) for c in candidates]

    payload = {
        "model": settings.SEARCH_ENGINE.get("COHERE_RERANK_MODEL") or "rerank-v3.5",
        "query": query,
        "documents": documents,
        "top_n": min(output_k, len(documents)),
    }
    url = (
        settings.SEARCH_ENGINE.get("COHERE_RERANK_BASE_URL") or "https://api.cohere.com/v2/rerank"
    )
    timeout = float(settings.SEARCH_ENGINE.get("COHERE_RERANK_TIMEOUT_S") or 10)

    try:
        import time as _time  # noqa: PLC0415

        import httpx  # noqa: PLC0415 — httpx is already an indirect dep

        # Rate-limit-aware retry. Trial keys are 10 calls/min (60s
        # rolling window); production keys are ~1000/min. Both can
        # transiently 429 under bursty load (e.g. a 39-case eval
        # batch). Retries honor `Retry-After` when Cohere sends it;
        # otherwise we sleep `COHERE_RERANK_RETRY_BACKOFF_S` (default
        # 15s) so a few back-to-back trial-key bursts naturally fall
        # back into the next window. Tunable via
        # `COHERE_RERANK_MAX_RETRIES` (default 3 = up to ~45s of
        # cumulative wait per failing call).
        max_retries = int(settings.SEARCH_ENGINE.get("COHERE_RERANK_MAX_RETRIES", 3))
        retry_backoff = float(settings.SEARCH_ENGINE.get("COHERE_RERANK_RETRY_BACKOFF_S", 15))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            attempts = 1
            while resp.status_code == 429 and attempts <= max_retries:
                wait_s = float(resp.headers.get("Retry-After") or retry_backoff)
                log.warning(
                    "Cohere rerank 429; sleeping %.1fs then retrying (attempt %d/%d)",
                    wait_s,
                    attempts,
                    max_retries,
                )
                _time.sleep(wait_s)
                resp = client.post(url, headers=headers, json=payload)
                attempts += 1

        if resp.status_code != 200:
            log.warning(
                "Cohere rerank returned %d: %s; falling back to pre-rerank order",
                resp.status_code,
                resp.text[:200],
            )
            return candidates[:output_k]
        data = resp.json()
    except Exception:  # noqa: BLE001 — surface as fallback, not a crash
        log.exception("Cohere rerank call failed; falling back to pre-rerank order")
        return candidates[:output_k]

    # v2 response shape: {"id": "...", "results": [{"index": 0,
    # "relevance_score": 0.93}, ...]}. Results are pre-sorted by
    # relevance_score descending; we honor that order.
    raw_results = data.get("results") or []
    indices: list[int] = []
    relevance: dict[int, float] = {}
    seen: set[int] = set()
    for r in raw_results:
        idx = r.get("index")
        if not isinstance(idx, int):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            indices.append(idx)
            try:
                relevance[idx] = float(r.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                relevance[idx] = 0.0

    if not indices:
        log.warning(
            "Cohere rerank returned no usable indices (%r); falling back to pre-rerank order",
            data,
        )
        return candidates[:output_k]

    # D2 score fusion: Cohere's relevance_score is a real, calibrated
    # cross-encoder relevance — the faithful "RRF score + cross-encoder
    # score" blend. Without fusion we honor Cohere's score-desc order.
    weight = _fusion_weight(query)
    if weight is not None:
        return _fuse_by_score(candidates, relevance, weight, output_k)

    return [candidates[i] for i in indices[:output_k]]


def _cohere_doc_text(entity: dict[str, Any]) -> str:
    """Flatten an entity to the single string Cohere reranks against.

    Same surface the LLM reranker sees (title + truncated snippet) so
    the A/B between the two providers compares like-for-like. Empty
    components are skipped so we don't ship blank documents that
    Cohere would reject.
    """
    title = (entity.get("title") or "").strip()
    snippet = (entity.get("snippet") or "").strip()
    snippet = (
        snippet.replace("<workspace_content>", "").replace("</workspace_content>", "").strip()
    )
    if len(snippet) > _SNIPPET_TRUNCATE:
        snippet = snippet[:_SNIPPET_TRUNCATE].rstrip() + "…"
    if title and snippet:
        return f"{title}\n{snippet}"
    return title or snippet or "(untitled)"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _build_user_prompt(query: str, candidates: list[dict[str, Any]]) -> str:
    """Build the numbered-list prompt the model sees."""
    lines: list[str] = [f"Query: {query}", "", "Candidates:"]
    for i, e in enumerate(candidates):
        eid = e.get("entity_id") or "?"
        title = (e.get("title") or "").strip()
        snippet = (e.get("snippet") or "").strip()
        # Strip the Phase-4 boundary tags if a snippet was wrapped on
        # its way out of `search_kb` — they're noise for the reranker
        # which already understands the candidates are workspace data.
        snippet = snippet.replace("<workspace_content>", "").replace("</workspace_content>", "")
        snippet = snippet.strip()
        if len(snippet) > _SNIPPET_TRUNCATE:
            snippet = snippet[:_SNIPPET_TRUNCATE].rstrip() + "…"
        lines.append(f"[{i}] {eid} | {title} | {snippet}")
    return "\n".join(lines)


# Match the first JSON array of integers in the model's response.
# Models sometimes wrap their answer in code fences or add a trailing
# period; this regex finds the array anywhere in the string.
_JSON_ARRAY_RE = re.compile(r"\[[\s\d,]*\]")


def _parse_indices(raw: str, *, valid_range: int) -> list[int] | None:
    """Extract a list of valid 0-based indices from the model's reply.

    Returns None if the response doesn't contain a valid array of
    integers, all-in-range and unique. An empty array `[]` is valid
    and means "no candidate is relevant".
    """
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    out: list[int] = []
    seen: set[int] = set()
    for x in parsed:
        if not isinstance(x, int):
            return None
        if x < 0 or x >= valid_range:
            return None
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
