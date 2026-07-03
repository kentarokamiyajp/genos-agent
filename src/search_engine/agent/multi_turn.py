"""Phase 3.5 — multi-turn context preparation.

Shared between the production /ask/ view and the eval runner so both
paths exercise the same prior-turn truncation / summarisation logic.

The decision tree, given `all_turns` (full session history of
(query, answer) pairs) and the `max_verbatim` window:

  * `rolling_summary == False` OR `len(all_turns) <= max_verbatim`:
    pass through the last `max_verbatim` turns verbatim, no summary.
    This is the current pre-3.5 behavior; flag-off keeps it exactly.

  * `rolling_summary == True` AND `len(all_turns) > max_verbatim`:
    keep the last `max_verbatim` turns verbatim AND summarise the
    earlier turns into a single short paragraph that the controller
    prepends to the messages list.

The summary is computed lazily — one LLM call per turn that triggers
the summary path. We do NOT persist it on the AgentSession (yet) — for
the MVP, recomputing per turn keeps the schema unchanged and the cost
visible. If turn N pays for a summary of turns 1..(N-3), turn N+1
pays for a (similar but distinct) summary of turns 1..(N-2). Worth
upgrading to incremental persisted summaries (Phase 5-ish) if/when
session lengths routinely exceed 6-7 turns in prod.
"""

from __future__ import annotations

import logging

from django.conf import settings

from origin.search_engine.llm import AgentMessage, get_model_client

log = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """\
You are summarising a multi-turn conversation between a user and an AI
workspace assistant. The summary will be re-injected as context for the
NEXT turn so the assistant can resolve references to earlier topics.

Produce ONE short paragraph (<= 120 words). Capture:
- TOPICS the user asked about (one phrase each).
- KEY ENTITIES mentioned by name (project names, task names, person
  names) — verbatim, so later turns can match them.
- The conversation ARC (e.g. "started with X, then narrowed to Y").

Skip greetings, meta-commentary, and the assistant's prose. Just the
substantive context. Write it as a third-person factual recap, not as
dialogue (do not say "the user said …" — just state the topics).
"""

_SUMMARY_USER_TEMPLATE = """\
Conversation so far ({n_turns} turns, oldest first):

{convo}

Now write the summary paragraph.
"""

# Hard cap on how much of each prior turn's text we feed into the
# summary call — defends against runaway prompt size when a prior
# answer is long-form markdown.
_PER_TURN_TEXT_CAP = 600


def build_prior_context(
    all_turns: list[tuple[str, str]],
    *,
    rolling_summary: bool | None = None,
    max_verbatim: int | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Decide which prior turns travel verbatim and (optionally) build a
    rolling summary of the rest.

    Returns:
        (verbatim_turns, summary_or_none)

    Args:
        all_turns: Full session history of (query, answer) pairs,
            chronologically oldest first.
        rolling_summary: Force the summary path on/off. Default None
            reads the `RAG_SESSION_ROLLING_SUMMARY` setting (so callers
            don't have to plumb it through).
        max_verbatim: Override the verbatim window. Default None reads
            `SESSION_MAX_PRIOR_TURNS` (so callers stay in sync with
            production).
    """
    if rolling_summary is None:
        rolling_summary = bool(settings.SEARCH_ENGINE.get("RAG_SESSION_ROLLING_SUMMARY", False))
    if max_verbatim is None:
        max_verbatim = int(settings.SEARCH_ENGINE.get("SESSION_MAX_PRIOR_TURNS", 3))

    if not all_turns:
        return [], None

    if not rolling_summary or len(all_turns) <= max_verbatim:
        return all_turns[-max_verbatim:], None

    to_summarise = all_turns[:-max_verbatim]
    verbatim = all_turns[-max_verbatim:]
    summary = _summarise_turns(to_summarise)
    return verbatim, summary


def _summarise_turns(turns: list[tuple[str, str]]) -> str | None:
    """Render `turns` into a short paragraph via one LLM call. Returns
    `None` on any failure — caller must treat None as "fall back to
    verbatim-only behavior" (i.e. the summary is best-effort).
    """
    if not turns:
        return None

    convo_lines: list[str] = []
    for i, (q, a) in enumerate(turns, start=1):
        q_text = (q or "").strip()
        a_text = (a or "").strip()
        if len(q_text) > _PER_TURN_TEXT_CAP:
            q_text = q_text[: _PER_TURN_TEXT_CAP - 1] + "…"
        if len(a_text) > _PER_TURN_TEXT_CAP:
            a_text = a_text[: _PER_TURN_TEXT_CAP - 1] + "…"
        convo_lines.append(f"Q{i}: {q_text}\nA{i}: {a_text}")
    convo_text = "\n\n".join(convo_lines)

    user_prompt = _SUMMARY_USER_TEMPLATE.format(
        n_turns=len(turns),
        convo=convo_text,
    )

    try:
        client = get_model_client()
        chunks: list[str] = []
        for text, _fcall in client.generate_step(
            messages=[AgentMessage(role="user", text=user_prompt)],
            tools=[],
            system_instruction=_SUMMARY_SYSTEM,
        ):
            if text:
                chunks.append(text)
        out = "".join(chunks).strip()
        return out or None
    except Exception:  # noqa: BLE001 — never break the agent loop on summary failure
        log.exception("Rolling-summary LLM call failed; falling back to no-summary")
        return None
