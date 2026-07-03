"""Chunker for the team-shared `spotlight_answer` lane.

Collects completed Spotlight answers (`AgentRun`) so a later teammate who asks
a similar question finds the past answer in Spotlight typeahead instead of the
agent re-deriving it. This is the team-shared sibling of `conversation_chunker`
(the per-user memory lane):

  * `entity_type="spotlight_answer"`, one chunk per collected run.
  * ACL = the INTERSECTION of every source's ACL (see
    `agent/source_visibility.py`) — the answer is visible only to users who
    could have seen ALL of its evidence. Answers built from any unclassifiable
    or single-person source are dropped (fail-closed). This matters because the
    answer *body* can quote a source, not just the chips.
  * Provenance: `related_entity_ids` + `answer_sources` (the SpotlightResult-
    shaped source dicts) + `answer_text` so the frontend can render the past
    answer with clickable source chips and inline citations.

v1 collects single-turn runs only (the first/only run in a session): a
follow-up turn can lean on a prior turn's private source its own steps don't
record, so gating it from its own sources alone would be unsound.

Indexed by the batch reindex (`ingest_all(entity_types=["spotlight_answer"])` /
`opensearch_reindex`), NOT on the answer path — zero added latency to the live
ask. Runs that no longer qualify (e.g. a cited project later turned private)
are purged on the next full reindex via an empty `EntityChunks` tombstone.

Known v1 limitation: the gate constrains the answer + its sources, but the
QUESTION text (stored in `title` / `search_text` / `snippet`) comes from
`run.query` and is NOT itself ACL-checked. The collected answer is only ever
visible to the source-audience intersection (never the whole team), so the
exposure is bounded — but a user who pastes private text into a question whose
answer happens to cite only broadly-shared sources would surface that question
text to that intersection. The dominant case is safe (the question's subject is
usually retrieved as a source, which narrows the intersection). Gating on the
question text too is future work.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator, Optional

from origin.search_engine.agent.source_visibility import shareable_acl_for_sources
from origin.search_engine.chunkers.base import (
    Chunk,
    EntityChunks,
    iso,
    make_snippet,
)
from origin.search_engine.models import AgentRun

# Inline citation tokens the agent embeds in answers (e.g. "[task:123]",
# "[chat:dm:uuid]"). Mirrors the frontend's strip pattern. We drop them from
# the user-facing snippet so the typeahead preview reads cleanly; the full
# `answer_text` keeps them so the "Previous answer" view can resolve each
# token to a clickable source chip.
_CITATION_TOKEN_RE = re.compile(r"\[(?:chat|task|note|project|todo):[^\]\s]+\]")

# Only clean completions are worth collecting (mirrors the conversation lane /
# the F2 sampler scope): rejected / step_cap / error runs are not durable.
_INDEXABLE_STATUS = "done"
# Bound the answer that feeds search_text/embedding (mirrors conversation lane).
_MAX_ANSWER_SEARCH_CHARS = 4000
# Bound the stored display copy. Generous — Spotlight answers are short — but
# capped so one runaway answer can't bloat the index document.
_MAX_ANSWER_DISPLAY_CHARS = 12000


def _is_first_turn(run: AgentRun) -> bool:
    """True if `run` is the only/first turn in its session.

    A follow-up turn shares its session with an earlier run; v1 does not
    collect those (their answer can rely on a prior turn's private source).
    A run with no session is a standalone ask — always single-turn.
    """
    if run.session_id is None:
        return True
    return not AgentRun.objects.filter(
        session_id=run.session_id, started_at__lt=run.started_at
    ).exists()


def iter_spotlight_answer_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    # Lazy import: `controller` pulls in the LLM client stack; keep it off this
    # module's import path (and out of any ingestion import cycle).
    from origin.search_engine.agent.controller import (
        _hydrate_task_display_ids,
        reconstruct_sources_for_run,
    )

    qs = AgentRun.objects.filter(status=_INDEXABLE_STATUS).exclude(final_answer_text="")
    if since is not None:
        qs = qs.filter(started_at__gte=since)
    qs = qs.order_by("started_at")

    for run in qs.iterator():
        entity_id = f"spotlight_answer:{run.run_id}"

        query = (run.query or "").strip()
        answer = (run.final_answer_text or "").strip()
        team_id = str(run.team_id or "")
        # Follow-up turns are never collected (and were never indexed), so skip
        # them outright rather than emitting a tombstone.
        if not query or not answer or not team_id or not _is_first_turn(run):
            continue

        sources = _hydrate_task_display_ids(reconstruct_sources_for_run(run))
        acl = shareable_acl_for_sources(sources)
        if acl is None:
            # Built from private / unclassifiable / single-person sources.
            # Emit a tombstone so any previously-indexed copy is purged.
            yield EntityChunks(entity_type="spotlight_answer", entity_id=entity_id, chunks=[])
            continue

        search_answer = (
            answer
            if len(answer) <= _MAX_ANSWER_SEARCH_CHARS
            else answer[:_MAX_ANSWER_SEARCH_CHARS] + "…"
        )
        display_answer = (
            answer
            if len(answer) <= _MAX_ANSWER_DISPLAY_CHARS
            else answer[:_MAX_ANSWER_DISPLAY_CHARS] + "…"
        )

        chunk = Chunk(
            chunk_id=entity_id,
            entity_type="spotlight_answer",
            entity_id=entity_id,
            chunk_type="spotlight_answer",
            team_id=team_id,
            acl_user_ids=acl,
            # The original question is the most useful one-line handle when the
            # answer surfaces as a typeahead result.
            title=query if len(query) <= 200 else query[:197] + "…",
            # search_text carries BOTH question and answer so a recall query
            # matches on either side.
            search_text=f"Q: {query}\nA: {search_answer}",
            snippet_text=make_snippet(_CITATION_TOKEN_RE.sub("", answer)),
            related_entity_ids=[s["entity_id"] for s in sources if s.get("entity_id")],
            # Provenance for the frontend "Previous answer" card.
            answer_text=display_answer,
            answer_sources=sources,
            created_at=iso(run.started_at),
            updated_at=iso(run.finished_at or run.started_at),
        )
        yield EntityChunks(entity_type="spotlight_answer", entity_id=entity_id, chunks=[chunk])
