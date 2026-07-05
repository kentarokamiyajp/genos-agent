"""Canonical NDJSON event-type vocabulary for the agent stream.

⚠️ KEEP IN SYNC — genos-frontend: src/services/agentEventNames.ts

This is the wire protocol of POST /api/v2/agent/ask/ and /decide/
(emitted by `agent/controller.py` and `agent_views.py`, consumed by the
frontend's `runNdjsonStream`). Each repo pins its own code to its copy
of this list: `origin/tests/test_agent_event_contract.py` asserts every
``{"type": ...}`` event literal the emitters produce is listed here (and
that nothing stale lingers); the frontend asserts it dispatches every
listed name to a handler. Adding or renaming an event on one side
without the other turns a silent production break (dead UI, stuck
"streaming…" spinner) into a red build.

Name-level tripwire only — field renames *inside* an event are not
covered. Contract & rationale: genos-docs
spotlight/SPOTLIGHT_AGENT_CHANGE_SAFETY.md §4.3.
"""

AGENT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "sources",
        "answer_delta",
        "done",
        "error",
        "tool_call_start",
        "tool_call_result",
        "tool_call_error",
        "tool_call_pending_approval",
    }
)

# Events after which the backend intentionally closes (or pauses) the
# stream. The frontend's `streamClosedCleanly` invariant — "a stream that
# ends without one of these is an error" — depends on this exact set, so
# a NEW terminal event is a frontend change too, not just a backend one.
AGENT_TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "done",
        "error",
        "tool_call_pending_approval",
    }
)
