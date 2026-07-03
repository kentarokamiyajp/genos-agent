"""Hybrid search service.

Pipeline:

    user query
        ├─ keyword search (BM25 over title/snippet/search_text)
        ├─ vector search  (k-NN over `embedding`)
        ↓
    Reciprocal Rank Fusion (RRF)
        ↓
    Group by `entity_type:entity_id`, take the best chunk per entity
        ↓
    Top-N results

Filters applied at OpenSearch query time:
  * team_id (mandatory tenant boundary)
  * acl_user_ids contains the requesting user_id
  * entity_types subset (optional)
  * updated_at range (optional)

Both keyword and vector queries return up to `pool_size` chunk hits
each (default 60). The wider pool gives RRF more material to fuse.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from django.conf import settings
from opensearchpy.exceptions import NotFoundError

from origin.search_engine.embeddings import embed_one
from origin.search_engine.opensearch_client import get_client, get_index_alias

log = logging.getLogger(__name__)


RRF_K = 60
DEFAULT_POOL_SIZE = 60
DEFAULT_LIMIT = 20


# Caller-mode dispatch. Each mode tunes the hybrid pipeline for a
# different consumer:
#
#   "typeahead"  → Spotlight UI Cmd-K. Sub-100 ms target. Smaller
#                  candidate pool, tight HNSW ef_search, precision-
#                  favoured chunk-type ranking (raw messages outrank
#                  LLM summaries because the user wants the literal
#                  hit, not an abstract).
#   "ai_search"  → agent's `search_knowledge_base` tool. Higher recall
#                  budget (more chunks to feed the LLM, looser HNSW),
#                  recall-favoured weights (LLM-curated summaries
#                  outrank raw concatenations because the agent reads
#                  abstracts better than walls of text).
#   "eval"      → offline `agent_eval --retrieval` harness. Wide pool, flat weights,
#                  no freshness boost — pure retrieval quality.
#
# Per-mode hyperparameters live in `_MODE_CONFIG` below so a tweak
# touches one dict, not three call sites.
SearchMode = Literal["typeahead", "ai_search", "eval"]

_MODE_CONFIG: dict[str, dict] = {
    "typeahead": {
        "pool_size": 20,
        "ef_search": 64,
        "apply_freshness": True,
        # Precision-favoured chunk-type weights. Raw chunks (literal
        # messages, sections, task titles) outrank LLM-curated abstracts
        # because the typeahead user types verbatim keywords ("Plausible",
        # "framer-motion") and wants the literal hit. Summaries are
        # demoted but kept so a question-shaped typeahead still returns
        # something useful.
        "chunk_type_weights": {
            "chat_message": 1.0,
            "task_title_content": 1.0,
            "milestone_title_content": 1.0,
            "note_section": 1.0,
            # A collected past answer that matches the typed question is highly
            # relevant — parity with raw chunks (not boosted above them, so the
            # live source still wins when both match equally).
            "spotlight_answer": 1.0,
            "task_content_chunk": 0.8,
            "task_comment": 0.7,
            "thread_summary": 0.6,
            "note_summary": 0.6,
            "chat_thread_window": 0.5,
        },
    },
    "ai_search": {
        "pool_size": 60,
        "ef_search": 128,
        "apply_freshness": True,
        # Recall-favoured chunk-type weights. LLM-curated summaries
        # outrank raw concatenations because the agent reads abstracts
        # better than walls of text. Per-message and per-section chunks
        # stay at parity 1.0 — the agent still wants the exact wording
        # when its question is targeted.
        "chunk_type_weights": {
            "thread_summary": 1.2,
            "note_summary": 1.2,
            "chat_message": 1.0,
            "note_section": 1.0,
            "task_title_content": 1.0,
            "milestone_title_content": 1.0,
            "task_content_chunk": 1.0,
            "task_comment": 1.0,
            "chat_thread_window": 0.8,
        },
    },
    "eval": {
        "pool_size": 100,
        "ef_search": 256,
        "apply_freshness": False,
        # Flat weights so retrieval-quality numbers reflect the
        # underlying RRF + BM25 + vector ranking, not the tuned
        # production weights above.
        "chunk_type_weights": {},
    },
}

# Default relevance threshold relative to the top result's RRF score.
# Anything below `top_score * MIN_SCORE_RATIO` is treated as a weak
# match and dropped, even if it would otherwise fit under `limit`. So
# a query with one strong hit and a long tail of near-noise returns
# just the strong hit, but a query with several near-tied hits returns
# all of them.
#
# Why a ratio instead of an absolute number: RRF scores are bounded
# above by 1/(RRF_K+1) ≈ 0.016 per lane (so ≤ 0.033 with both lanes),
# but the *useful* range depends on how many lanes fired and how the
# query distributes across them. A fixed absolute threshold would
# misbehave when only one lane is active (e.g. when OPENAI_API_KEY is
# missing and vector search is skipped).
DEFAULT_MIN_SCORE_RATIO = 0.5

# Absolute minimum: anything below this is noise regardless of the top
# score. Cuts the "vector lane finds something for every query, even
# gibberish" failure mode — for queries like "quantum photosynthesis
# xylophone marauder" the vector lane returns plausible-looking but
# semantically empty matches with RRF scores ≈ 0.027–0.033 (chunk
# pool has uniform low similarity, no real signal). Setting this above
# that band kills the noise.
#
# A/B (2026-05) on the 39-case retrieval suite found the safe range
# is [0.035, 0.050] — within it `gibberish_returns_few_or_nothing`
# moves FAIL→PASS with no regressions; at ≥ 0.060, legitimate weak
# paraphrase queries (`paraphrase_q3_planning_artifacts`,
# `scope_website_redesign_top`) start losing. 0.040 is the middle of
# the safe band. Tunable per call AND per deploy via
# `SEARCH_ENGINE["RAG_MIN_SCORE"]`.
DEFAULT_MIN_SCORE = 0.040


def search(
    *,
    query: str,
    team_id: str,
    user_id: str,
    entity_types: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    pool_size: Optional[int] = None,
    use_vector: bool = True,
    min_score_ratio: float = DEFAULT_MIN_SCORE_RATIO,
    min_score: float = DEFAULT_MIN_SCORE,
    for_agent: bool = False,
    max_chunks_per_entity: int = 3,
    rewrite: bool = False,
    mode: Optional[SearchMode] = None,
) -> dict:
    """Run a hybrid search and return entity-grouped results.

    Args:
        query: user-supplied query string.
        team_id: tenant — required.
        user_id: requesting user — used for ACL filter.
        entity_types: subset, e.g. ["chat","note"]. Default: all.
        date_from/date_to: ISO 8601 strings (compared against `updated_at`).
        limit: max number of entity-level results to return (after
            relevance filtering).
        pool_size: raw chunk pool size per search lane.
        use_vector: if False, skip vector lane (keyword-only fallback —
            useful when no OPENAI_API_KEY is set).
        min_score_ratio: drop results whose RRF score is below
            `top_score * min_score_ratio`. Pass 0 to disable. Default
            0.5 — meaning we only return results within ~half the top
            result's confidence.
        min_score: absolute floor on the RRF score. Pass 0 to disable.
            Default trims pure-noise matches (single lane, rank ≥ 30).
        for_agent: if True, return a richer shape suitable for stuffing
            into an LLM prompt: includes `search_text` (the full chunk
            text) and up to `max_chunks_per_entity` matched chunks per
            entity. The UI-facing shape (snippet only, one chunk per
            entity) is the default to keep wire size small.
        max_chunks_per_entity: when `for_agent=True`, cap on how many
            chunks per entity are returned. Default 3 — keeps prompt
            size bounded but gives the LLM more than just the snippet.
        rewrite: Phase 10 — expand the query into multiple variants via
            the configured `ModelClient` before retrieval, then fuse
            results across all variants. Default `False` so callers
            don't accidentally pay the LLM round-trip. The agent's
            `search_knowledge_base` tool reads
            `SEARCH_ENGINE["RAG_USE_QUERY_REWRITE"]` and passes it
            through; the Spotlight typeahead endpoint never opts in
            (would cost an LLM call per keystroke).
        mode: caller-mode selector — one of "typeahead" / "ai_search"
            / "eval". Picks the right hyperparameter set from
            `_MODE_CONFIG` (pool_size, HNSW ef_search, freshness on/off,
            chunk-type weights). When None, infers from `for_agent`:
            True → ai_search, False → typeahead. Pinning `pool_size`
            explicitly still wins so callers that know their workload
            can override.
    """
    if not query or not query.strip():
        return {"query": query, "results": []}

    # Mode resolution. Infer from `for_agent` if caller didn't pin it
    # — keeps existing call sites working with no signature changes.
    if mode is None:
        mode = "ai_search" if for_agent else "typeahead"
    mode_cfg = _MODE_CONFIG.get(mode, _MODE_CONFIG["typeahead"])
    if pool_size is None:
        pool_size = mode_cfg["pool_size"]
    ef_search = mode_cfg["ef_search"]
    apply_freshness_flag = mode_cfg["apply_freshness"]
    chunk_type_weights: dict[str, float] = mode_cfg.get("chunk_type_weights") or {}

    # Settings-level override hooks for the threshold knobs. When the
    # caller didn't explicitly pin a value (i.e. left the kwarg at its
    # Python default), settings can override at deploy time without
    # touching code. Caller-pinned values always win — useful for
    # tests that need a specific threshold regardless of env.
    if min_score_ratio == DEFAULT_MIN_SCORE_RATIO:
        min_score_ratio = float(
            settings.SEARCH_ENGINE.get("RAG_MIN_SCORE_RATIO", DEFAULT_MIN_SCORE_RATIO)
        )
    if min_score == DEFAULT_MIN_SCORE:
        min_score = float(settings.SEARCH_ENGINE.get("RAG_MIN_SCORE", DEFAULT_MIN_SCORE))

    client = get_client()
    index = get_index_alias()

    base_filter = _build_filter(team_id, user_id, entity_types, date_from, date_to, mode=mode)

    # --- Phase 10: query rewriting (optional) ---
    # `variants` always starts with the original query; the rewriter
    # adds N alternative phrasings. With rewriting off we get a one-
    # element list and the loop below collapses to the pre-Phase-10
    # behavior exactly.
    if rewrite:
        from origin.search_engine.query_rewriter import rewrite_query  # noqa: PLC0415

        num_variants = int(settings.SEARCH_ENGINE.get("RAG_REWRITE_NUM_VARIANTS", 3))
        variants = rewrite_query(query, num_variants=num_variants)
    else:
        variants = [query]

    # --- Run keyword + vector for each variant, then merge ---
    # We RRF-fuse each variant independently (so two-lane scoring stays
    # well-calibrated per variant) and SUM the per-variant scores at
    # the chunk level. Chunks that surface for multiple variants get
    # extra weight, which is exactly the boost rewriting should give.
    fused = _multi_variant_fuse(
        variants=variants,
        client=client,
        index=index,
        base_filter=base_filter,
        pool_size=pool_size,
        use_vector=use_vector,
        for_agent=for_agent,
        ef_search=ef_search,
    )

    # --- v2: chunk-type-aware reweighting ---
    # Multiply each chunk's RRF score by its chunk_type's weight
    # (see `_MODE_CONFIG[mode].chunk_type_weights`). Lets the
    # typeahead mode favour raw chunks (literal-keyword hits) and the
    # ai_search mode favour LLM-curated summaries without changing the
    # underlying schema or requiring a reindex.
    if chunk_type_weights:
        for hit in fused:
            ct = (hit.get("source") or {}).get("chunk_type")
            weight = chunk_type_weights.get(ct, 1.0) if ct else 1.0
            if weight != 1.0:
                hit["score"] *= weight

    # --- Phase 6: freshness multiplier + text-hash dedup ---
    # Both are no-ops when their settings are at the disable values
    # (half_life=0, dedup_by_hash=false), so the default path matches
    # the pre-Phase-6 behavior exactly. `apply_freshness_flag` is the
    # per-mode kill switch — `mode="eval"` disables it to keep the
    # offline retrieval-quality harness deterministic.
    half_life = float(settings.SEARCH_ENGINE.get("RAG_FRESHNESS_HALF_LIFE_DAYS", 0) or 0)
    if apply_freshness_flag and half_life > 0:
        fused = _apply_freshness(fused, half_life_days=half_life)
    if settings.SEARCH_ENGINE.get("RAG_DEDUP_BY_HASH"):
        fused = _dedup_by_text_hash(fused)
    # Freshness can re-order; re-sort once before grouping so the
    # "first occurrence wins" rule in `_group_by_entity` still picks
    # the best chunk per entity by the new score.
    fused.sort(key=lambda x: x["score"], reverse=True)

    # --- Group by entity ---
    grouped = _group_by_entity(
        fused, for_agent=for_agent, max_chunks_per_entity=max_chunks_per_entity
    )

    # --- Sort, apply relevance threshold, truncate to limit. ---
    grouped.sort(key=lambda x: x["score"], reverse=True)
    pre_threshold = grouped  # keep the un-cut ranking for the floor restore
    grouped = _apply_relevance_threshold(grouped, min_score_ratio, min_score)

    # --- Phase 1.2: per-entity-type representation floor (flag-gated) ---
    # Off by default. When on, ensures the result set contains the
    # top-ranked entity of each type that existed in the pre-threshold
    # ranking. Counteracts the "task crowds out a strong chat" failure
    # where threshold ratio cuts a legitimate #2/#3 hit just because
    # the #1 hit is a different type with much higher score.
    if settings.SEARCH_ENGINE.get("RAG_REPRESENTATION_FLOOR") and pre_threshold:
        grouped = _apply_representation_floor(grouped, pre_threshold)

    # --- Phase 6: optional rerank stage (LLM judge or Cohere, flag-gated) ---
    # Off by default. When on, dispatched via RAG_RERANKER_PROVIDER
    # to either the LLM judge or Cohere v2 Rerank. Reranker module
    # falls back to pre-rerank order on any error.
    #
    # `RAG_RERANK_LOCK_TOP_N` (default 0 = disabled) lets us protect
    # the top-N RRF hits from being reshuffled by the semantic
    # reranker. A/B on this suite showed that any semantic reranker
    # applied to the *whole* result set wins on paraphrase queries but
    # regresses on exact-phrase queries (RRF's already-confident top
    # hits get displaced by semantically-richer-but-less-precise
    # candidates). Locking the top-N from RRF preserves those wins
    # while still letting the reranker fix the noisy tail.
    # Mode guard: NEVER rerank in typeahead mode. The reranker makes an LLM
    # call (seconds), which is incompatible with typeahead's sub-100 ms Cmd-K
    # budget. Reranking/fusion is an ai_search (agent path) + eval (offline
    # measurement) concern only — so flipping RAG_USE_RERANKER on by default
    # (Q2.1, backed by measured +0.118 recall on the agent path) doesn't
    # silently tax the as-you-type surface.
    if settings.SEARCH_ENGINE.get("RAG_USE_RERANKER") and grouped and mode != "typeahead":
        from origin.search_engine.reranker import rerank  # noqa: PLC0415

        input_k = int(settings.SEARCH_ENGINE.get("RAG_RERANK_INPUT_K", 20))
        output_k = int(settings.SEARCH_ENGINE.get("RAG_RERANK_OUTPUT_K", 10))
        lock_top_n = int(settings.SEARCH_ENGINE.get("RAG_RERANK_LOCK_TOP_N", 0))
        lock_top_n = max(0, min(lock_top_n, len(grouped)))

        # Score fusion (D2) is the SOFT form of the hard top-N lock —
        # it blends RRF + reranker relevance across the whole candidate
        # set rather than freezing a head. When fusion is on, bypass the
        # lock and fuse the whole set (the reranker module does the blend
        # internally; see RAG_RERANK_FUSION).
        fusion_on = settings.SEARCH_ENGINE.get("RAG_RERANK_FUSION", False)

        if lock_top_n > 0 and not fusion_on and len(grouped) > lock_top_n:
            # Lock the top-N RRF hits; rerank only the tail. Final
            # output = locked head + reranked tail (deduped, capped
            # at output_k).
            head = grouped[:lock_top_n]
            tail = grouped[lock_top_n:]
            # Effective input_k for the tail-only rerank — the locked
            # head doesn't count toward the model's input budget.
            tail_input_k = max(0, input_k - lock_top_n)
            tail_output_k = max(0, min(output_k, limit) - lock_top_n)
            reranked_tail = (
                rerank(
                    query=query,
                    entities=tail,
                    input_k=tail_input_k,
                    output_k=tail_output_k,
                )
                if tail_output_k > 0 and tail_input_k > 0
                else []
            )
            grouped = head + reranked_tail
        else:
            grouped = rerank(
                query=query,
                entities=grouped,
                input_k=input_k,
                output_k=min(output_k, limit),
            )

    # --- GraphRAG: fuse the relationship graph into ranking (Q2.4 / A1). ---
    # After hybrid retrieval, pull in one-hop TaskDependency neighbors of the
    # top task hits and inject them with a decayed score, so a relational
    # query ("what's blocked by the framer-motion spike?") surfaces the
    # graph-related task even when it shares no text with the query. Skipped
    # on typeahead (its sub-100 ms budget can't afford the extra DB+OS round
    # trip); flag-gated. ACL is automatic — neighbors are fetched through the
    # same acl_user_ids filter as everything else.
    if settings.SEARCH_ENGINE.get("RAG_GRAPH_EXPANSION") and mode != "typeahead" and grouped:
        grouped = _graph_expand(
            grouped,
            team_id=team_id,
            user_id=user_id,
            for_agent=for_agent,
            client=client,
            index=index,
        )

    # --- Final pass: friendly chat titles. ---
    # OpenSearch stores a viewer-agnostic placeholder for chats ("DM 9")
    # because a DM's name depends on who's looking. Resolve here so
    # every search consumer (typeahead, agent) sees the same friendly
    # name (partner / group / project name).
    from origin.search_engine.friendly_titles import apply_friendly_titles  # noqa: PLC0415

    final = grouped[:limit]
    apply_friendly_titles(final, user_id)

    return {"query": query, "results": final}


def _multi_variant_fuse(
    *,
    variants: list[str],
    client,
    index: str,
    base_filter: list[dict],
    pool_size: int,
    use_vector: bool,
    for_agent: bool,
    ef_search: int,
) -> list[dict]:
    """Run keyword + vector for each variant and merge into one ranked list.

    Per-variant: `_run_keyword` + `_run_vector` (if enabled) → `_rrf_fuse`.
    Across variants: chunk-level WEIGHTED score summation. The original
    query (index 0 in `variants`) contributes at a higher weight than
    LLM-generated rewrites because it carries the user's actual intent;
    rewrites are aids that can drift on-topic-but-off-meaning. Without
    this weighting, an exact-phrase match against the original (e.g.
    "competitor analysis" → task titled "Competitor analysis") gets
    crowded out by RRF noise from variant-only hits.

    Weight is `RAG_REWRITE_ORIGINAL_WEIGHT` (default 2.0); each
    variant contributes weight 1.0. With weight 1.0 the behavior is
    byte-identical to the un-weighted Phase-10 path. Single-variant
    case (`len(variants) == 1`) is unaffected by weighting.
    """
    original_weight = float(settings.SEARCH_ENGINE.get("RAG_REWRITE_ORIGINAL_WEIGHT", 2.0))

    chunks_by_id: dict[str, dict] = {}
    for idx, variant in enumerate(variants):
        weight = original_weight if idx == 0 else 1.0

        keyword_hits = _run_keyword(
            client, index, variant, base_filter, pool_size, for_agent=for_agent
        )
        vector_hits: list[dict] = []
        if use_vector:
            try:
                qvec = embed_one(variant)
                vector_hits = _run_vector(
                    client,
                    index,
                    qvec,
                    base_filter,
                    pool_size,
                    for_agent=for_agent,
                    ef_search=ef_search,
                )
            except Exception as e:  # noqa: BLE001 — degrade to keyword-only for this variant
                log.warning(
                    "Vector search failed for variant %r, keyword-only: %s", variant[:80], e
                )

        variant_fused = _rrf_fuse(keyword_hits, vector_hits)
        for hit in variant_fused:
            cid = hit["chunk_id"]
            weighted_score = hit["score"] * weight
            existing = chunks_by_id.get(cid)
            if existing is None:
                # First time we see this chunk — keep the dict, but
                # store the weighted score so subsequent merges are on
                # equal footing.
                copied = dict(hit)
                copied["score"] = weighted_score
                chunks_by_id[cid] = copied
                continue
            # Same chunk surfaced for a previous variant. Add the
            # weighted RRF contribution and keep best (lowest) lane
            # ranks for the UI's debug fields.
            existing["score"] += weighted_score
            existing["keyword_rank"] = _min_rank(
                existing.get("keyword_rank"), hit.get("keyword_rank")
            )
            existing["vector_rank"] = _min_rank(
                existing.get("vector_rank"), hit.get("vector_rank")
            )
    return sorted(chunks_by_id.values(), key=lambda x: x["score"], reverse=True)


def _min_rank(a, b):
    """Lower rank is better; pick the smaller of two values, ignoring None."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _apply_freshness(hits: list[dict], *, half_life_days: float) -> list[dict]:
    """Multiply each hit's score by an exponential decay on `updated_at`.

    Formula: `score *= exp(-age_days / half_life_days)`. Result: a
    same-day update keeps its score; one half-life old loses half its
    score; chunks with no `updated_at` are left alone.

    Operates in place on the fused-chunk list and returns it for
    convenience.
    """
    now = datetime.now(timezone.utc)
    for hit in hits:
        ts_str = (hit.get("source") or {}).get("updated_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        except ValueError:
            # Unparseable timestamp → don't penalize.
            continue
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        hit["score"] *= math.exp(-age_days / half_life_days)
    return hits


def _dedup_by_text_hash(hits: list[dict]) -> list[dict]:
    """Drop duplicate chunks by `text_hash`, keeping the highest score.

    Catches the case where identical content is indexed under multiple
    chunks — e.g. a note that quotes a chat message verbatim. Hits
    without a `text_hash` are passed through unchanged (we never merge
    two distinct chunks just because they happen to lack a hash).
    """
    seen: dict[str, dict] = {}
    out: list[dict] = []
    for hit in hits:
        text_hash = (hit.get("source") or {}).get("text_hash")
        if not text_hash:
            out.append(hit)
            continue
        existing = seen.get(text_hash)
        if existing is None:
            seen[text_hash] = hit
            out.append(hit)
        elif hit["score"] > existing["score"]:
            # Replace in-place: bump the score onto the kept hit so we
            # don't have to re-walk `out`.
            existing["score"] = hit["score"]
            existing["keyword_rank"] = hit.get("keyword_rank") or existing.get("keyword_rank")
            existing["vector_rank"] = hit.get("vector_rank") or existing.get("vector_rank")
    return out


def _apply_relevance_threshold(
    grouped: list[dict], min_score_ratio: float, min_score: float
) -> list[dict]:
    """Drop results that are weak in absolute or relative terms.

    `grouped` must already be sorted by score desc.

    Adaptive guard: when the threshold would leave fewer than
    `RAG_THRESHOLD_MIN_SURVIVORS` entities, we treat that as evidence
    the candidate set is already tight (small or low-confidence corpus
    for this query) and skip the cut entirely. The original bug this
    addresses: a query that surfaced one strong DM + two weaker hits
    saw all three cut because the strong DM made the ratio floor too
    high — leaving the user with zero results instead of three useful
    ones. Default min_survivors=3; setting 0 disables the guard.
    """
    if not grouped:
        return grouped
    top_score = grouped[0]["score"]
    relative_floor = top_score * min_score_ratio if min_score_ratio > 0 else 0.0
    floor = max(relative_floor, min_score)
    if floor <= 0:
        return grouped
    survivors = [g for g in grouped if g["score"] >= floor]

    min_survivors = int(settings.SEARCH_ENGINE.get("RAG_THRESHOLD_MIN_SURVIVORS", 3))
    if 0 < min_survivors and len(survivors) < min_survivors:
        # Keep the top-`min_survivors` un-cut so the consumer always
        # sees a usable set. They were ranked highest pre-threshold;
        # their absolute scores being low just means this query was
        # narrow, not that the matches are wrong.
        return grouped[:min_survivors]
    return survivors


def _apply_representation_floor(survivors: list[dict], pre_threshold: list[dict]) -> list[dict]:
    """Guarantee each entity type that had ANY match survives the cut.

    Walks `pre_threshold` (the un-cut, score-sorted ranking) for each
    entity type. If a type is missing from `survivors`, the highest-
    ranked entity of that type is appended back.

    Preserves the score order of `survivors`; restored entries land at
    the tail (they were already below the threshold so they shouldn't
    outrank the legitimate survivors, but they're better than zero
    representation for that type).
    """
    if not pre_threshold:
        return survivors

    present_types = {e.get("entity_type") for e in survivors}
    seen_ids = {(e.get("entity_type"), e.get("entity_id")) for e in survivors}
    out = list(survivors)
    for entity in pre_threshold:
        etype = entity.get("entity_type")
        if etype in present_types:
            continue
        key = (etype, entity.get("entity_id"))
        if key in seen_ids:
            continue
        out.append(entity)
        present_types.add(etype)
        seen_ids.add(key)
    return out


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _build_filter(
    team_id: str,
    user_id: str,
    entity_types: Optional[list[str]],
    date_from: Optional[str],
    date_to: Optional[str],
    mode: Optional[SearchMode] = None,
) -> list[dict]:
    filt: list[dict] = [
        {"term": {"team_id": team_id}},
        {"term": {"acl_user_ids": user_id}},
    ]
    if entity_types:
        filt.append({"terms": {"entity_type": entity_types}})
    else:
        # Default-search exclusions. Both are reachable only when a caller
        # opts in explicitly via `entity_types`:
        #   * conversation     — the per-user Q2.3 memory lane (private; the
        #                         `search_past_conversations` tool opts in).
        #   * spotlight_answer  — collected team answers. Surfaced in Spotlight
        #                         typeahead, but kept OUT of agent grounding
        #                         (ai_search / eval) so a past answer can't feed
        #                         back into a new answer (answer→grounding loop).
        excluded = ["conversation"]
        if mode != "typeahead":
            excluded.append("spotlight_answer")
        filt.append({"bool": {"must_not": [{"term": {"entity_type": et}} for et in excluded]}})
    if date_from or date_to:
        rng: dict = {}
        if date_from:
            rng["gte"] = date_from
        if date_to:
            rng["lte"] = date_to
        filt.append({"range": {"updated_at": rng}})
    return filt


# Trailing `:msg:<id>` on a chat_message chunk's id. Used to surface
# the matched message id to the frontend so Spotlight can deep-link to
# the exact message bubble. Matches both main-channel
# (`chat:dm:<uuid>:msg:<uuid>`) and in-thread
# (`chat:dm:<uuid>:thread:<uuid>:msg:<uuid>`) message chunks; anchor
# chunks (`...:anchor:<uuid>`) and thread-window chunks (`...:window`)
# don't match, which is what we want — an anchor id refers to the
# thread-root message which doesn't map cleanly to a reply URL, and
# windows aggregate many messages.
#
# The id is the v3 `Message.id` UUID ([0-9a-fA-F-]); the legacy numeric
# form is a subset of this class, so the pattern stays back-compatible.
_CHAT_MSG_ID_RE = re.compile(r":msg:([0-9a-fA-F-]+)$")


def _extract_chat_message_id(chunk_id: str | None, chunk_type: str | None) -> str | None:
    """Pull the message id from a chat_message chunk's id, or None.
    See `_CHAT_MSG_ID_RE` for the formats this recognises.
    """
    if chunk_type != "chat_message" or not chunk_id:
        return None
    m = _CHAT_MSG_ID_RE.search(chunk_id)
    return m.group(1) if m else None


_HIGHLIGHT_PRE = "\x02"
_HIGHLIGHT_POST = "\x03"
# Matches one analyzer-marked term inside the highlight response, e.g.
# "\x02running\x03". Control chars are used instead of `<em>` so we
# can't collide with literal `<em>` substrings in user-generated text.
_HIGHLIGHT_TERM_RE = re.compile(rf"{_HIGHLIGHT_PRE}(.*?){_HIGHLIGHT_POST}", re.DOTALL)


def _extract_matched_terms(highlight: Optional[dict]) -> list[str]:
    """Pull the unique analyzer-matched terms (stemmed/synonym forms
    included) out of an OpenSearch hit's `highlight` block. Returned
    lowercased and sorted longest-first so the frontend regex alternation
    gives longer tokens priority over their prefixes.
    """
    if not highlight:
        return []
    seen: set[str] = set()
    for fragments in highlight.values():
        if not isinstance(fragments, list):
            continue
        for frag in fragments:
            if not frag:
                continue
            for term in _HIGHLIGHT_TERM_RE.findall(frag):
                cleaned = term.strip().lower()
                if cleaned:
                    seen.add(cleaned)
    return sorted(seen, key=len, reverse=True)


def _run_keyword(
    client, index: str, query: str, base_filter: list[dict], size: int, *, for_agent: bool = False
) -> list[dict]:
    body = {
        "size": size,
        # `track_total_hits: false` — RRF only needs the top-N; counting
        # exact total hits is wasted work on every query.
        "track_total_hits": False,
        # Allowlist projection: `_source_fields` already excludes the
        # `embedding` blob, but we add an explicit `excludes` as defence
        # in depth in case `_source_fields` ever drifts.
        "_source": {
            "includes": _source_fields(for_agent=for_agent),
            "excludes": ["embedding"],
        },
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        # Field-level BM25 boosts. Title is the
                        # densest signal — short, intentional, often
                        # the verbatim query for "find me this thing"
                        # asks. Snippet_text is the next-densest
                        # (entity-level highlight). Search_text
                        # carries the full chunk body. v2 added two
                        # subfields:
                        #   title.prefix   — edge n-gram, wins on
                        #                    1-3 char prefixes
                        #                    ("fra" → "framer-motion")
                        #                    even before BM25 partial
                        #                    matching kicks in.
                        #   search_text.en — English-stemmed copy of
                        #                    the body; recovers
                        #                    conjugation variants
                        #                    ("ruling/ruled/rules").
                        # Base `search_text` (standard analyzer) stays
                        # for exact-phrase matching; the .en subfield
                        # is the recall path.
                        # v3 multilingual subfields (analysis-icu +
                        # analysis-kuromoji). `.icu` segments CJK into
                        # words + folds accents for all 7 languages;
                        # `.ja` adds Japanese base-form/inflection
                        # recall (走った/走って → 走る). `.ja` is weighted
                        # just above `.icu` (higher-precision JA lane);
                        # all sit under the base/`standard` boosts so an
                        # exact-phrase hit still wins ties.
                        "fields": [
                            f"title^{int(settings.SEARCH_ENGINE.get('RAG_BM25_TITLE_BOOST', 4))}",
                            "title.prefix^4",
                            "title.icu^3",
                            "title.ja^3.2",
                            "title.icu_prefix^3",
                            f"snippet_text^{int(settings.SEARCH_ENGINE.get('RAG_BM25_SNIPPET_BOOST', 2))}",
                            "snippet_text.icu^1.6",
                            "snippet_text.ja^1.7",
                            "search_text",
                            "search_text.en^0.8",
                            "search_text.icu^0.9",
                            "search_text.ja^1.0",
                        ],
                        "type": "best_fields",
                    }
                },
                "filter": base_filter,
            }
        },
        # Highlight lets the frontend bold analyzer-matched terms
        # (stemming, synonyms) — not just literal user-typed words.
        # `number_of_fragments: 0` returns the whole field with markers
        # rather than fragments; we only consume the marked term list,
        # not the marked text itself.
        #
        # v2: also highlight `title.prefix` + `search_text.en` so the
        # frontend's matched-term list picks up stemmed/prefix hits.
        "highlight": {
            "pre_tags": [_HIGHLIGHT_PRE],
            "post_tags": [_HIGHLIGHT_POST],
            "fields": {
                "title": {"number_of_fragments": 0},
                "title.prefix": {"number_of_fragments": 0},
                "title.icu": {"number_of_fragments": 0},
                "title.ja": {"number_of_fragments": 0},
                "title.icu_prefix": {"number_of_fragments": 0},
                "snippet_text": {"number_of_fragments": 0},
                "snippet_text.icu": {"number_of_fragments": 0},
                "snippet_text.ja": {"number_of_fragments": 0},
                "search_text": {"number_of_fragments": 0},
                "search_text.en": {"number_of_fragments": 0},
                "search_text.icu": {"number_of_fragments": 0},
                "search_text.ja": {"number_of_fragments": 0},
            },
        },
    }
    try:
        resp = client.search(index=index, body=body)
    except NotFoundError:
        return []
    return list(resp.get("hits", {}).get("hits", []))


def _run_vector(
    client,
    index: str,
    qvec: list[float],
    base_filter: list[dict],
    size: int,
    *,
    for_agent: bool = False,
    ef_search: int = 128,
) -> list[dict]:
    # `ef_search` controls how many HNSW candidates the engine inspects
    # before returning `k`. Lucene's default is 512 — overkill at MVP
    # corpus size. Per-mode tuning (see `_MODE_CONFIG`) drops this to
    # 64/128 for typeahead/ai_search and saves real wall-clock per query.
    body = {
        "size": size,
        "track_total_hits": False,
        "_source": {
            "includes": _source_fields(for_agent=for_agent),
            "excludes": ["embedding"],
        },
        "query": {
            "bool": {
                "must": {
                    "knn": {
                        "embedding": {
                            "vector": qvec,
                            "k": size,
                            "method_parameters": {"ef_search": ef_search},
                        }
                    }
                },
                "filter": base_filter,
            }
        },
    }
    try:
        resp = client.search(index=index, body=body)
    except NotFoundError:
        return []
    return list(resp.get("hits", {}).get("hits", []))


def _source_fields(*, for_agent: bool = False) -> list[str]:
    fields = [
        "chunk_id",
        "entity_type",
        "entity_id",
        "chunk_type",
        "title",
        "snippet_text",
        "chat_type",
        "chat_id",
        "thread_id",
        "task_id",
        "note_id",
        "note_type",
        "project_id",
        "related_entity_ids",
        "updated_at",
        "created_at",
        # spotlight_answer lane — stored-only provenance projected so the
        # frontend "Previous answer" card can render the past answer and its
        # clickable source chips. Absent (and so omitted) on every other lane.
        "answer_text",
        "answer_sources",
        # Phase 6 — pulled into projection so `_dedup_by_text_hash` can
        # collapse near-duplicates. SHA-256 of the chunk's search_text
        # written by the chunker; identical text → identical hash.
        "text_hash",
    ]
    if for_agent:
        # The full chunk text — used as LLM grounding context. Excluded
        # from the UI-facing shape to keep wire size small (the UI only
        # needs `snippet_text`).
        fields.append("search_text")
    return fields


def _rrf_fuse(keyword_hits: list[dict], vector_hits: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: combine two ranked chunk-hit lists.

    Returns a list of
    `{chunk_id, source, score, keyword_rank, vector_rank, matched_terms}`
    sorted by RRF score. `matched_terms` is the analyzer-aware token list
    pulled from the keyword hit's highlight response (empty for
    vector-only matches).
    """
    by_chunk: dict[str, dict] = {}

    for rank, hit in enumerate(keyword_hits, start=1):
        cid = hit["_id"]
        by_chunk.setdefault(
            cid,
            {
                "chunk_id": cid,
                "source": hit["_source"],
                "score": 0.0,
                "keyword_rank": None,
                "vector_rank": None,
                "matched_terms": [],
            },
        )
        by_chunk[cid]["score"] += 1.0 / (RRF_K + rank)
        by_chunk[cid]["keyword_rank"] = rank
        by_chunk[cid]["source"] = hit["_source"]
        # Only keyword hits carry highlights — overwrite unconditionally.
        by_chunk[cid]["matched_terms"] = _extract_matched_terms(hit.get("highlight"))

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit["_id"]
        by_chunk.setdefault(
            cid,
            {
                "chunk_id": cid,
                "source": hit["_source"],
                "score": 0.0,
                "keyword_rank": None,
                "vector_rank": None,
                "matched_terms": [],
            },
        )
        by_chunk[cid]["score"] += 1.0 / (RRF_K + rank)
        by_chunk[cid]["vector_rank"] = rank
        by_chunk[cid]["source"] = hit["_source"]

    return sorted(by_chunk.values(), key=lambda x: x["score"], reverse=True)


def _group_by_entity(
    fused_chunks: list[dict],
    *,
    for_agent: bool = False,
    max_chunks_per_entity: int = 3,
) -> list[dict]:
    """Collapse chunk-level hits into entity-level rows.

    Per entity we keep:
      * highest chunk score → entity score
      * all chunk types that matched
      * the highest-ranked chunk's snippet

    When `for_agent=True`, also attach a `chunks` list with up to
    `max_chunks_per_entity` matched chunks (each with `chunk_id`,
    `chunk_type`, and `text`) so the caller can stuff full chunk text
    into an LLM prompt instead of only the short snippet.
    """
    by_entity: dict[tuple[str, str], dict] = {}
    for c in fused_chunks:
        src = c["source"]
        key = (src.get("entity_type"), src.get("entity_id"))
        existing = by_entity.get(key)
        if existing is None:
            entry = {
                "entity_type": src.get("entity_type"),
                "entity_id": src.get("entity_id"),
                "title": src.get("title"),
                "best_matched_chunk_id": c["chunk_id"],
                "matched_chunk_types": [src.get("chunk_type")] if src.get("chunk_type") else [],
                "snippet": src.get("snippet_text"),
                "score": c["score"],
                "keyword_rank": c["keyword_rank"],
                "vector_rank": c["vector_rank"],
                "updated_at": src.get("updated_at"),
                "chat_type": src.get("chat_type"),
                "chat_id": src.get("chat_id"),
                "thread_id": src.get("thread_id"),
                "task_id": src.get("task_id"),
                "note_id": src.get("note_id"),
                "note_type": src.get("note_type"),
                "project_id": src.get("project_id"),
                "related_entity_ids": src.get("related_entity_ids") or [],
                "matched_terms": list(c.get("matched_terms") or []),
                # spotlight_answer lane provenance — present only on
                # spotlight_answer hits (None elsewhere). Lets the frontend
                # render the stored answer + clickable source chips.
                "answer_text": src.get("answer_text"),
                "answer_sources": src.get("answer_sources"),
                # Surfaced for chat_message chunks so the frontend can
                # deep-link straight to the matched message bubble.
                # None for chat_thread_window / anchor chunks and for
                # non-chat entities — frontend treats that as "land on
                # the chat/thread, no specific message focus".
                "message_id": _extract_chat_message_id(c["chunk_id"], src.get("chunk_type")),
            }
            if for_agent:
                entry["chunks"] = [_chunk_for_agent(c)]
            by_entity[key] = entry
        else:
            chunk_type = src.get("chunk_type")
            if chunk_type and chunk_type not in existing["matched_chunk_types"]:
                existing["matched_chunk_types"].append(chunk_type)
            # Backfill the deep-link target: if the top (highest-scoring)
            # chunk for this entity was a thread-window / anchor (no single
            # message id), a lower-ranked per-message chunk can still supply
            # one so the chip deep-links to a concrete bubble instead of
            # landing at the chat/thread top.
            if not existing.get("message_id"):
                mid = _extract_chat_message_id(c["chunk_id"], src.get("chunk_type"))
                if mid:
                    existing["message_id"] = mid
            if for_agent and len(existing.get("chunks", [])) < max_chunks_per_entity:
                existing["chunks"].append(_chunk_for_agent(c))
            # Merge analyzer-matched terms across chunks of the same
            # entity — a lower-ranked chunk may have surfaced a stemmed
            # form the top chunk missed.
            new_terms = c.get("matched_terms") or []
            if new_terms:
                seen = set(existing["matched_terms"])
                for t in new_terms:
                    if t not in seen:
                        existing["matched_terms"].append(t)
                        seen.add(t)
            # Keep the highest score — `fused_chunks` is already sorted
            # by score desc, so the first occurrence wins.
    return list(by_entity.values())


def _chunk_for_agent(c: dict) -> dict:
    src = c["source"]
    return {
        "chunk_id": c["chunk_id"],
        "chunk_type": src.get("chunk_type"),
        "text": src.get("search_text") or src.get("snippet_text") or "",
        "score": c["score"],
    }


def _graph_expand(
    grouped: list[dict],
    *,
    team_id: str,
    user_id: str,
    for_agent: bool,
    client,
    index: str,
) -> list[dict]:
    """GraphRAG ranking-fusion (A1 / Q2.4).

    Walk one hop of the `TaskDependency` graph out from the top retrieved
    TASK hits and inject the graph-adjacent tasks into the result set with a
    score decayed from the source hit's score. This surfaces a related task
    the lexical/vector lanes can't reach (the relation lives in a FK table,
    not in the text) — e.g. "what's blocked by the framer-motion spike?"
    surfaces the blocked Hero build even though that task never mentions
    framer-motion.

    Bounded + cheap: at most `RAG_GRAPH_MAX_SOURCES` source tasks → ONE
    `TaskDependency` query (both directions) → ONE OpenSearch fetch for the
    neighbours, capped at `RAG_GRAPH_MAX_NEIGHBORS`. ACL is inherited: the
    neighbour fetch goes through `_build_filter` (team + acl_user_ids), so a
    task the user can't see is never injected. Neighbours already present in
    `grouped` are skipped (no dup, no double-count).
    """
    from django.db.models import Q  # noqa: PLC0415

    from origin.models.task.task_models import TaskDependency  # noqa: PLC0415

    se = settings.SEARCH_ENGINE
    weight = float(se.get("RAG_GRAPH_WEIGHT", 0.6))
    max_sources = int(se.get("RAG_GRAPH_MAX_SOURCES", 5))
    max_neighbors = int(se.get("RAG_GRAPH_MAX_NEIGHBORS", 5))

    # Source task ids = the top task hits (graph edges are task↔task). Only
    # expand from a LEXICALLY-anchored hit (keyword_rank present): a real
    # topical query hits the keyword lane, whereas gibberish / vague-vector
    # noise matches nothing lexically — so this gate stops graph expansion
    # from inflating the result set on a query that should return ~nothing
    # (e.g. the `gibberish_returns_few_or_nothing` case), while still firing
    # on genuine relational queries.
    src_score: dict[int, float] = {}
    present_ids: set = {e.get("entity_id") for e in grouped}
    for e in grouped:
        if e.get("entity_type") != "task" or not e.get("task_id"):
            continue
        if e.get("keyword_rank") is None:
            continue
        try:
            tid = int(e["task_id"])
        except (TypeError, ValueError):
            continue
        src_score.setdefault(tid, float(e.get("score") or 0.0))
        if len(src_score) >= max_sources:
            break
    if not src_score:
        return grouped

    ids = list(src_score)
    edges = (
        TaskDependency.objects.filter(team_id=team_id)
        .filter(Q(blocker_task_id__in=ids) | Q(blocked_task_id__in=ids))
        .values_list("blocker_task_id", "blocked_task_id")
    )

    # Neighbour task id → best (max) decayed score across the sources that
    # reach it. A neighbour already in `grouped` is left alone.
    neigh_score: dict[int, float] = {}
    for blocker, blocked in edges:
        for end, other in ((blocker, blocked), (blocked, blocker)):
            if end in src_score and other not in src_score and f"task:{other}" not in present_ids:
                neigh_score[other] = max(neigh_score.get(other, 0.0), src_score[end] * weight)
    if not neigh_score:
        return grouped

    neigh_ids = sorted(neigh_score, key=lambda t: -neigh_score[t])[:max_neighbors]
    entity_ids = [f"task:{t}" for t in neigh_ids]

    # ACL-filtered fetch of the neighbour task chunks (same filter as search).
    flt = _build_filter(team_id, user_id, ["task"], None, None)
    flt.append({"terms": {"entity_id": entity_ids}})
    try:
        resp = client.search(
            index=index,
            body={"size": max(len(entity_ids) * 4, 8), "query": {"bool": {"filter": flt}}},
            _source_excludes=["embedding"],
        )
        hits = resp.get("hits", {}).get("hits", [])
    except Exception:  # noqa: BLE001 — graph expansion is best-effort, never fatal
        log.exception("graph expansion: neighbour fetch failed")
        return grouped
    if not hits:
        return grouped

    fused: list[dict] = []
    for h in hits:
        src = h.get("_source") or {}
        try:
            tid = int(src.get("task_id"))
        except (TypeError, ValueError):
            continue
        fused.append(
            {
                "chunk_id": src.get("chunk_id") or h.get("_id"),
                "source": src,
                "score": neigh_score.get(tid, 0.0),
                "keyword_rank": None,
                "vector_rank": None,
                "matched_terms": [],
            }
        )
    if not fused:
        return grouped

    neighbours = _group_by_entity(fused, for_agent=for_agent)
    for e in neighbours:
        try:
            tid = int(e.get("task_id"))
        except (TypeError, ValueError):
            tid = None
        e["score"] = neigh_score.get(tid, 0.0) if tid is not None else 0.0
        # Mark provenance so the chip-row / agent can tell a graph-pulled
        # result from a lexical/vector hit.
        e["matched_terms"] = list(
            dict.fromkeys((e.get("matched_terms") or []) + ["graph:dependency"])
        )
        e["graph_related"] = True

    grouped = grouped + neighbours
    grouped.sort(key=lambda e: float(e.get("score") or 0.0), reverse=True)
    return grouped
