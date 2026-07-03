"""Phase 10 — LLM query rewriter.

Expands a user query into N alternative phrasings before running
retrieval. Each variant takes a turn through the keyword + vector
lanes, and the results are RRF-fused across all variants. Chunks
that surface for multiple variants get extra weight, so the more
"agreed-upon" a result is, the higher it ranks.

Flag-gated via `SEARCH_ENGINE["RAG_USE_QUERY_REWRITE"]` (enabled by
default on the agent path; never fires on the typeahead path).
Reuses the Phase-5 `ModelClient` abstraction so it works on both
Gemini and Claude.

Graceful degradation: any error path (LLM failure, malformed
output, empty variants list) silently returns `[query]` so the
caller's pipeline runs exactly as if rewriting was disabled.
"""

from __future__ import annotations

import json
import logging
import re

from django.conf import settings

from origin.search_engine.llm import AgentMessage, get_model_client

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a search query rewriter for a workspace search engine. The
workspace contains the user's chats, tasks, and notes. Given a query,
produce alternative phrasings that retrieve the same intent but use
different words — synonyms, paraphrases, related domain terms.

Rules:
  - Reply with ONLY a JSON array of strings, like ["foo bar", "baz qux"].
  - Do NOT include the user's original query — only the variants.
  - Each variant is a standalone search query, not a sentence describing
    the query.
  - Variants must stay on the same topic. Don't drift to a different
    question.
  - Write every variant in the SAME language and script as the user's
    query (e.g. a Japanese query gets Japanese variants, not English).
    The corpus is indexed in the user's language, so off-language
    variants retrieve nothing and only add noise.
  - Prefer short, keyword-style variants (3–6 words) over long
    natural-language ones.
  - Do not include prose, explanations, code fences, or markdown.
"""


def rewrite_query(query: str, *, num_variants: int = 3) -> list[str]:
    """Return `[query, *variants]` — the original first, then expansions.

    Args:
        query:        the user's raw query.
        num_variants: how many LLM-generated variants to request. The
                      caller controls the budget — more variants =
                      more OpenSearch round-trips + more embeddings.

    The original query is always first in the returned list (we never
    drop it). On any failure path the function returns `[query]`, so
    the caller can iterate the result without special-casing the
    feature flag.
    """
    if not query or not query.strip():
        return []
    if num_variants <= 0:
        return [query]

    user_prompt = f"Query: {query}\n\nReturn {num_variants} variants."
    msgs = [AgentMessage(role="user", text=user_prompt)]

    try:
        client = get_model_client()
        # Optional fast-model override (RAG_REWRITE_MODEL). The rewrite
        # task — emit a few keyword variants — is trivial, so operators
        # can run it on a fast model while the synthesis model stays
        # heavy. Mirrors the reranker's RAG_RERANKER_MODEL override;
        # empty/unset → None → the client's default model.
        model_override = settings.SEARCH_ENGINE.get("RAG_REWRITE_MODEL") or None
        chunks: list[str] = []
        for text, fc in client.generate_step(
            messages=msgs,
            tools=[],
            system_instruction=_SYSTEM_PROMPT,
            model_override=model_override,
        ):
            if text:
                chunks.append(text)
            if fc is not None:
                log.warning(
                    "Query rewriter unexpectedly got a function call (name=%s) — ignoring",
                    fc.name,
                )
        raw = "".join(chunks).strip()
    except Exception:  # noqa: BLE001 — surface as fallback, not a crash
        log.exception("Query rewriter LLM call failed; using original query only")
        return [query]

    variants = _parse_variants(raw)
    if not variants:
        log.warning(
            "Query rewriter returned unparseable output (%r); using original query only",
            raw[:200],
        )
        return [query]

    # Deduplicate (case-insensitive) while preserving order. The
    # original is always first so it wins any tie with a variant that
    # happens to repeat it.
    seen: set[str] = {query.strip().lower()}
    out: list[str] = [query]
    for v in variants[:num_variants]:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

# Match the first JSON array of strings anywhere in the model's reply.
# `[.*?]` is non-greedy so a fenced response with code blocks before the
# array still picks up the array itself. DOTALL allows newlines inside.
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_variants(raw: str) -> list[str]:
    """Best-effort extraction of a JSON string array from the model output."""
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for x in parsed:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out
