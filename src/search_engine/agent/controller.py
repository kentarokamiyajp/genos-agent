"""Multi-step agent loop, with Phase 7 pause/resume for write tools.

Two entry points share the same per-step loop body:

  * `run_agent(query, ctx, emit, run_id=...)` — fresh run from a user
    query. Returns `None` on a clean finish, or a dict
    `{"paused": True, "approval_token": UUID, ...}` when the loop hit
    a `requires_approval` tool. The caller (view layer) is expected
    to write the token back onto `AgentRun.pending_approval_token`
    and flip `AgentRun.status` to `"awaiting_approval"`.

  * `resume_agent(run, decision, ctx, emit)` — resume a paused run.
    `decision` is `"approve"` or `"reject"`. Reconstructs the
    `messages` list from persisted `AgentStep` rows, executes (or
    rejects) the pending tool, and continues the loop. Same return
    shape as `run_agent` (could pause again on a subsequent write
    tool, though current tools don't chain that way).

Event types emitted (full NDJSON protocol):

  tool_call_start              read-only tool dispatch
  tool_call_result             read-only tool success
  tool_call_error              tool error (incl. user-rejected writes)
  tool_call_pending_approval   write tool — paused, awaiting user
  sources                      citation chips (after search calls)
  answer_delta                 streaming text from the final answer
  done                         final answer delivered
  error                        fatal mid-stream
"""

from __future__ import annotations

import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from uuid import UUID

from django.conf import settings
from django.db import connections

from origin.search_engine.agent import tool_cache
from origin.search_engine.agent.abstention import ABSTAIN_MESSAGE, is_abstention
from origin.search_engine.agent.citation_resolver import resolve_unresolved_citations
from origin.search_engine.agent.prompts import (
    AGENT_CRITIQUE_RETRIEVAL_DIRECTIVE,
    AGENT_SELF_CRITIQUE_PROMPT_TEMPLATE,
    AGENT_SELF_CRITIQUE_SYSTEM,
    AGENT_SYSTEM_PROMPT,
)
from origin.search_engine.agent.tools import REGISTRY, ToolContext, ToolError
from origin.search_engine.llm import (
    AgentMessage,
    FunctionCall,
    ToolDeclaration,
    get_model_client,
)
from origin.search_engine.llm.choice import _server_default_choice, get_llm_choice
from origin.search_engine.models import AgentRun, AgentStep

log = logging.getLogger(__name__)

# Marker stored in AgentStep.summary while a write tool is awaiting the
# user's decision. The resume path uses it to locate the pending row.
PENDING_APPROVAL_MARKER = "awaiting_approval"

# Decision strings accepted by `resume_agent`.
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"


def _persist_step(run_id: UUID | None, **fields: Any) -> AgentStep | None:
    """Best-effort write of one `AgentStep` row.

    Observability must NEVER break the user-facing path — if the DB
    insert fails for any reason, log it and move on. The agent stream
    completes regardless. Returns the saved row (or None if persistence
    is disabled / failed) so the pause path can update it later.
    """
    if run_id is None:
        return None
    try:
        return AgentStep.objects.create(run_id=run_id, **fields)
    except Exception:  # noqa: BLE001 — must not fail the response stream
        log.exception("Failed to persist AgentStep for run %s", run_id)
        return None


# One-shot latch so the kill-switch state is logged once per worker
# process (visible near startup) instead of spamming every request.
_KILLSWITCH_LOGGED = False


def _operator_disabled_tools() -> set[str]:
    """Resolve the `AGENT_DISABLED_TOOLS` ops kill-switch (+ log once).

    FAIL-OPEN: unset/empty disables nothing — env vars are per-service /
    per-environment config, so a service that misses the var must run at
    full capability rather than silently losing tools (see settings.py
    and SPOTLIGHT_AGENT_CHANGE_SAFETY.md §4.4). Unknown names disable
    nothing and are logged at ERROR: the operator believes something is
    switched off that isn't — exactly a typo's failure mode.
    """
    global _KILLSWITCH_LOGGED
    configured = settings.SEARCH_ENGINE.get("AGENT_DISABLED_TOOLS") or frozenset()
    if not configured:
        return set()
    known = set(configured) & set(REGISTRY)
    if not _KILLSWITCH_LOGGED:
        _KILLSWITCH_LOGGED = True
        log.warning(
            "AGENT_DISABLED_TOOLS active — tools hidden from the agent: %s",
            sorted(known),
        )
        unknown = set(configured) - known
        if unknown:
            log.error(
                "AGENT_DISABLED_TOOLS names unknown tool(s) %s — probable "
                "typo; they disable nothing",
                sorted(unknown),
            )
    return known


def _build_tool_declarations(
    disabled_tools: set[str] | None = None,
) -> list[ToolDeclaration]:
    """Translate each registered Tool into a provider-neutral declaration.

    Tools whose name appears in `disabled_tools` are omitted from the
    list, so the model never even sees them as callable. Used to honour
    the frontend "Web search" toggle (filters out `search_web`) and §4.5
    tool subsetting; the `AGENT_DISABLED_TOOLS` ops kill-switch is
    unioned in here — the single choke point every declaration build
    passes through. Note this hides tools from NEW model turns; a write
    tool already paused for approval still resumes if the user approves.
    """
    disabled = (disabled_tools or set()) | _operator_disabled_tools()
    return [
        ToolDeclaration(
            name=t.name,
            description=t.description,
            parameters_schema=t.parameters_schema,
        )
        for t in REGISTRY.values()
        if t.name not in disabled
    ]


# Per-context tool subsetting (RAG_TOOL_SUBSETTING — §4.5). Peripheral
# tool families are dropped from the declared tool list when the query
# shows no keyword signal for them. Families are derived from tool NAMES
# (robust to new tools joining a family); keywords use word boundaries so
# short tokens ("pr", "i") don't match inside unrelated words ("project",
# "list"). Core task/note/chat/project/analytics tools belong to no
# family and are never dropped.
_PERIPHERAL_FAMILY_KEYWORDS: dict[str, "re.Pattern[str]"] = {
    "pr": re.compile(
        r"\b(pr|prs|pull requests?|pull-requests?|github|merge|merged|commit|commits|"
        r"code review|diff)\b"
    ),
    "calendar": re.compile(
        r"\b(calendar|schedule|scheduling|meeting|meetings|event|events|"
        r"appointment|availability|agenda)\b"
    ),
    "todo": re.compile(r"\b(todo|todos|to-do|to do|checklist)\b"),
    "me": re.compile(r"\b(my|me|mine|i|i'm|im)\b|assigned to me|do i|am i|should i"),
}


def _tool_family(name: str) -> str | None:
    """Map a tool name to its peripheral family, or None for a core tool
    (core tools are never subset out)."""
    if name == "fetch_pr" or name.startswith("list_pr_"):
        return "pr"
    if "calendar" in name:
        return "calendar"
    if "todo" in name:
        return "todo"
    if name.startswith("get_my_") or name.startswith("list_my_"):
        return "me"
    return None


def _irrelevant_tool_families(query: str) -> set[str]:
    """Tool names to disable for `query` under RAG_TOOL_SUBSETTING.

    A peripheral family is excluded only when the query contains NONE of
    its keywords. Errs toward keeping: an over-broad keyword match keeps a
    possibly-unneeded family (harmless); a missed match would drop a
    needed family (the failure mode — minimised by conservative keyword
    lists and the one-shot caveat documented on the setting). Pure (no
    I/O) — unit-testable.
    """
    q = (query or "").lower()
    triggered = {fam for fam, rx in _PERIPHERAL_FAMILY_KEYWORDS.items() if rx.search(q)}
    return {
        t.name
        for t in REGISTRY.values()
        if (fam := _tool_family(t.name)) is not None and fam not in triggered
    }


# Tool arguments that are raw DB primary keys. The agent emits these
# verbatim in `tool_call_start` / `tool_call_pending_approval` events,
# where they'd surface in the UI's approval card and activity strip as
# meaningless numbers ("project_id: 46"). `_friendly_arguments` swaps
# them for human-readable labels before emission. The raw values stay
# in the persisted `AgentStep.arguments_json` row so the resume path
# still re-runs the tool with the canonical primary key.
def _resolve_task_display_id(raw: Any) -> str | None:
    """Look up `TaskMaster.display_id` ("WRD-5") for a raw task primary key."""
    try:
        tid = int(raw)
    except (TypeError, ValueError):
        return None
    from origin.models.task.task_models import TaskMaster  # noqa: PLC0415

    t = TaskMaster.objects.select_related("project").filter(task_id=tid).first()
    return t.display_id if t else None


def _resolve_project_name(raw: Any) -> str | None:
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    from origin.models.project.prj_models import ProjectMaster  # noqa: PLC0415

    return (
        ProjectMaster.objects.filter(project_id=pid).values_list("project_name", flat=True).first()
    )


def _resolve_user_name(raw: Any) -> str | None:
    if not raw:
        return None
    from origin.models.common.user_models import CustomUser  # noqa: PLC0415

    return CustomUser.objects.filter(id=str(raw)).values_list("username", flat=True).first()


def _resolve_todo_item_title(raw: Any) -> str | None:
    """Look up `ToDoItem.title` for a raw item primary key so the
    approval card surfaces the human-readable todo instead of "73".
    Returns None when the id doesn't resolve (e.g. a non-todo tool
    happens to use an `item_id` arg too — the raw value is shown).
    """
    try:
        tid = int(raw)
    except (TypeError, ValueError):
        return None
    from origin.models.chat.todo_models import ToDoItem  # noqa: PLC0415

    return ToDoItem.objects.filter(item_id=tid).values_list("title", flat=True).first()


def _resolve_milestone_title(raw: Any) -> str | None:
    try:
        mid = int(raw)
    except (TypeError, ValueError):
        return None
    from origin.models.task.milestone_models import MilestoneMaster  # noqa: PLC0415

    return (
        MilestoneMaster.objects.filter(milestone_id=mid).values_list("title", flat=True).first()
    )


def _resolve_note_folder_name(raw: Any) -> str | None:
    try:
        fid = int(raw)
    except (TypeError, ValueError):
        return None
    from origin.models.note.personal_note_models import PersonalNoteFolder  # noqa: PLC0415

    return PersonalNoteFolder.objects.filter(folder_id=fid).values_list("name", flat=True).first()


# Argument-key → resolver. Resolvers return None on a miss so we fall
# back to the raw value (the user sees the ID rather than a blank).
_FRIENDLY_ARG_RESOLVERS: dict[str, Callable[[Any], str | None]] = {
    "task_id": _resolve_task_display_id,
    "project_id": _resolve_project_name,
    "assignee_id": _resolve_user_name,
    "reporter_id": _resolve_user_name,
    "new_assignee_id": _resolve_user_name,
    "item_id": _resolve_todo_item_title,
    "parent_item_id": _resolve_todo_item_title,
    "milestone_id": _resolve_milestone_title,
    "existing_milestone_id": _resolve_milestone_title,
    "parent_task_id": _resolve_task_display_id,
    # Only the note tools take `folder_id` (My-Notes sidebar folders).
    "folder_id": _resolve_note_folder_name,
}


def _friendly_task_plan_arguments(out: dict[str, Any]) -> None:
    """In-place nested enrichment for `create_task_plan` arguments.

    The flat resolver pass above only touches top-level keys; a plan's
    assignees live inside `tasks[i].assignee_id` and
    `milestone.assignee_ids`. Resolve them all with ONE batched user
    query so the approval card shows usernames, not UUIDs. Structure is
    otherwise preserved — the frontend's structured preview renders the
    same shape the model proposed.
    """
    from origin.models.common.user_models import CustomUser  # noqa: PLC0415

    tasks = out.get("tasks") if isinstance(out.get("tasks"), list) else []
    milestone = out.get("milestone") if isinstance(out.get("milestone"), dict) else None

    ids: set[str] = set()
    for t in tasks:
        if isinstance(t, dict) and t.get("assignee_id"):
            ids.add(str(t["assignee_id"]))
    if milestone:
        for uid in milestone.get("assignee_ids") or []:
            if uid:
                ids.add(str(uid))
    if not ids:
        return

    names = {
        str(u_id): username
        for u_id, username in CustomUser.objects.filter(id__in=list(ids)).values_list(
            "id", "username"
        )
    }
    new_tasks = []
    for t in tasks:
        if isinstance(t, dict) and t.get("assignee_id"):
            t = {**t, "assignee_id": names.get(str(t["assignee_id"]), t["assignee_id"])}
        new_tasks.append(t)
    out["tasks"] = new_tasks
    if milestone and milestone.get("assignee_ids"):
        out["milestone"] = {
            **milestone,
            "assignee_ids": [
                names.get(str(uid), uid) for uid in milestone["assignee_ids"] if uid
            ],
        }


def _friendly_bulk_update_arguments(out: dict[str, Any]) -> None:
    """In-place nested enrichment for `update_tasks_bulk` arguments.

    Each update row gains `display_id` / `title` and a `current` snapshot
    of the fields the tool can change (one batched query for the whole
    batch), so the approval card can render a true old→new diff table.
    The proposed values and `task_id` stay untouched — the persisted
    `arguments_json` is raw anyway; this only shapes the wire event.
    """
    updates = out.get("updates")
    if not isinstance(updates, list) or not updates:
        return

    from origin.models.task.task_models import TaskMaster  # noqa: PLC0415

    ids: list[int] = []
    for row in updates:
        if isinstance(row, dict):
            try:
                ids.append(int(row.get("task_id")))
            except (TypeError, ValueError):
                continue
    if not ids:
        return

    by_id = {
        t.task_id: t
        for t in TaskMaster.objects.select_related("project").filter(task_id__in=ids)
    }
    new_updates = []
    for row in updates:
        if isinstance(row, dict):
            try:
                task = by_id.get(int(row.get("task_id")))
            except (TypeError, ValueError):
                task = None
            if task is not None:
                row = {
                    **row,
                    "display_id": task.display_id,
                    "title": task.title,
                    "current": {
                        "priority": task.priority,
                        "effort_level": task.effort_level,
                        "status": task.status,
                        "due_date": task.due_date.isoformat() if task.due_date else None,
                    },
                }
        new_updates.append(row)
    out["updates"] = new_updates


def _friendly_update_note_arguments(out: dict[str, Any]) -> None:
    """In-place nested enrichment for `update_note` arguments.

    ADDS `note_title` (resolved from the note tables) so the approval
    card can say which note is being edited. `note_id` itself is left
    numeric — unlike the flat resolvers this must not replace the key,
    because the preview needs both the label and the raw id."""
    ntype = str(out.get("note_type") or "").lower()
    try:
        nid = int(out.get("note_id"))
    except (TypeError, ValueError):
        return
    from origin.models.note.personal_note_models import PersonalNoteMaster  # noqa: PLC0415
    from origin.models.note.task_note_models import TaskNoteMaster  # noqa: PLC0415

    model = {"personal": PersonalNoteMaster, "task": TaskNoteMaster}.get(ntype)
    if model is None:
        return
    title = model.objects.filter(note_id=nid).values_list("title", flat=True).first()
    if title:
        out["note_title"] = title


# Tool-name → nested enrichment pass, applied after the flat resolvers.
_FRIENDLY_NESTED_ENRICHERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "create_task_plan": _friendly_task_plan_arguments,
    "update_tasks_bulk": _friendly_bulk_update_arguments,
    "update_note": _friendly_update_note_arguments,
}


def _friendly_arguments(args: dict[str, Any], tool_name: str | None = None) -> dict[str, Any]:
    """Return a copy of `args` with raw IDs replaced by human labels.

    Applied only at the wire-event boundary (the approval card + tool
    progress strip render this). The persisted `arguments_json` keeps
    the canonical primary keys so the resume path re-runs the tool
    correctly. `tool_name` selects an optional nested enrichment pass
    for composite tools whose IDs live below the top level.
    """
    out: dict[str, Any] = {}
    for key, value in args.items():
        resolver = _FRIENDLY_ARG_RESOLVERS.get(key)
        if resolver is not None:
            try:
                friendly = resolver(value)
            except Exception:  # noqa: BLE001 — labels never break the loop
                log.exception("Friendly-arg lookup failed for %s=%r", key, value)
                friendly = None
            out[key] = friendly if friendly is not None else value
        else:
            out[key] = value

    enricher = _FRIENDLY_NESTED_ENRICHERS.get(tool_name or "")
    if enricher is not None:
        try:
            enricher(out)
        except Exception:  # noqa: BLE001 — labels never break the loop
            log.exception("Nested friendly-arg enrichment failed for %s", tool_name)
    return out


def _coerce_signature(raw: Any) -> bytes | None:
    """Normalise a persisted thought_signature back to `bytes | None`.

    `models.BinaryField` may surface as `bytes`, `memoryview`, or
    `None` depending on the DB driver. The Gemini SDK's `Part`
    constructor expects `bytes`, so coerce explicitly. Empty buffers
    become None — an empty signature would still be rejected by the
    API, so treating it as missing avoids sending a malformed echo.
    """
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        return raw or None
    # Anything else (e.g. str under SQLite quirks) — coerce defensively.
    try:
        return bytes(raw) or None
    except (TypeError, ValueError):
        return None


def _user_turn(query: str) -> AgentMessage:
    return AgentMessage(role="user", text=query)


def _assistant_function_call_turn(function_call: FunctionCall) -> AgentMessage:
    return AgentMessage(role="assistant", function_call=function_call)


def _function_response_turn(name: str, response: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        role="tool_response",
        function_response_name=name,
        function_response=response,
    )


_WORKSPACE_OPEN = "<workspace_content>\n"
_WORKSPACE_CLOSE = "\n</workspace_content>"


def _strip_workspace_marker(s: str | None) -> str | None:
    """Reverse of `wrap_workspace_content` for UI-bound snippet text."""
    if not s:
        return s
    if s.startswith(_WORKSPACE_OPEN) and s.endswith(_WORKSPACE_CLOSE):
        return s[len(_WORKSPACE_OPEN) : -len(_WORKSPACE_CLOSE)]
    return s


def _ui_source_for_match(match: dict[str, Any]) -> dict[str, Any]:
    """Shape a search-tool match into the UI's `sources` event payload.

    Mirrors the `SpotlightResult` shape returned by `/api/v2/search/`
    so the frontend can hand the source chip directly to the same
    `handleSpotlightSelect` router that the search-result rows use.
    Two fields are essential for routing parity:

      * `message_id` — lets a chat citation deep-link to the exact
        bubble that matched (not just the chat/thread).
      * `related_entity_ids` — fallback the frontend reads when chunks
        pre-date direct `task_id` / `chat_*` fields on note rows. Older
        chat-note / task-note chunks only carry their parent entity in
        this list, so dropping it breaks routing for unupgraded data.
    """
    return {
        "entity_type": match.get("entity_type"),
        "entity_id": match.get("entity_id"),
        "title": match.get("title"),
        "snippet": _strip_workspace_marker(match.get("snippet")),
        "chat_type": match.get("chat_type"),
        "chat_id": match.get("chat_id"),
        "thread_id": match.get("thread_id"),
        "message_id": match.get("message_id"),
        "task_id": match.get("task_id"),
        # Human-readable task ID ("<project.code>-<project_task_number>",
        # e.g. "PRJ-42"). Hydrated by `_hydrate_task_display_ids` after
        # the source list is built — the OpenSearch index doesn't carry it.
        "task_display_id": None,
        "note_id": match.get("note_id"),
        "note_type": match.get("note_type"),
        "project_id": match.get("project_id"),
        "matched_chunk_types": list(match.get("matched_chunk_types") or []),
        "matched_terms": list(match.get("matched_terms") or []),
        "related_entity_ids": list(match.get("related_entity_ids") or []),
        "updated_at": match.get("updated_at"),
        # These ranking fields are search-result-only; the agent never
        # ranks sources itself. Defaults keep the shape uniform so the
        # frontend doesn't have to branch on agent-vs-search origin.
        "score": 0.0,
        "keyword_rank": None,
        "vector_rank": None,
    }


from origin.search_engine.friendly_titles import (
    apply_friendly_titles as _resolve_chat_titles,
)


def _apply_friendly_titles(
    sources: list[dict[str, Any]], ctx: ToolContext
) -> list[dict[str, Any]]:
    """Replace placeholder chat titles ('DM 9') with viewer-friendly names.

    Thin adapter over the shared `friendly_titles.apply_friendly_titles`
    helper — kept so the in-loop call signature stays terse and so
    structured-tool sources (which don't go through `search()`) still
    get title resolution before chip emission.
    """
    return _resolve_chat_titles(sources, ctx.user_id)


def _hydrate_task_display_ids(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill `task_display_id` for task sources that don't already have one.

    `_task_source` (called from structured tools like list_tasks /
    fetch_task) sets display_id directly because the tool result already
    carries it. `_ui_source_for_match` (search-knowledge-base path) does
    NOT — the OpenSearch index stores only the raw task_id. We resolve
    those missing ones here with one batched DB query.
    """
    missing_ids: list[int] = []
    for src in sources:
        if src.get("entity_type") != "task" or src.get("task_display_id"):
            continue
        raw = src.get("task_id")
        if raw is None:
            continue
        try:
            missing_ids.append(int(raw))
        except (TypeError, ValueError):
            continue

    if not missing_ids:
        return sources

    from origin.models.task.task_models import TaskMaster

    by_id: dict[int, str] = {}
    for t in TaskMaster.objects.select_related("project").filter(task_id__in=missing_ids):
        by_id[t.task_id] = t.display_id

    for src in sources:
        if src.get("entity_type") != "task" or src.get("task_display_id"):
            continue
        try:
            tid = int(src.get("task_id"))
        except (TypeError, ValueError):
            continue
        if tid in by_id:
            src["task_display_id"] = by_id[tid]
    return sources


def _blank_source(entity_type: str, entity_id: str) -> dict[str, Any]:
    """Skeleton source dict; structured-tool helpers fill in the type-specific fields."""
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "title": None,
        "snippet": None,
        "chat_type": None,
        "chat_id": None,
        "thread_id": None,
        "message_id": None,
        "task_id": None,
        "task_display_id": None,
        "note_id": None,
        "note_type": None,
        "project_id": None,
        "matched_chunk_types": [],
        "matched_terms": [],
        "related_entity_ids": [],
        "updated_at": None,
        "score": 0.0,
        "keyword_rank": None,
        "vector_rank": None,
    }


def _task_source(
    task_id: Any, title: Any, project_id: Any, display_id: Any = None
) -> dict[str, Any]:
    s = _blank_source("task", f"task:{task_id}")
    s["title"] = title or ""
    s["task_id"] = str(task_id) if task_id is not None else None
    s["task_display_id"] = display_id or None
    s["project_id"] = str(project_id) if project_id is not None else None
    return s


def _project_source(project_id: Any, project_name: Any) -> dict[str, Any]:
    s = _blank_source("project", f"project:{project_id}")
    s["title"] = project_name or ""
    s["project_id"] = str(project_id) if project_id is not None else None
    return s


def _chat_source(
    chat_type: Any,
    chat_id: Any,
    thread_id: Any = None,
    title: Any = None,
) -> dict[str, Any]:
    # Chunker convention: entity_id has no leading "chat:" prefix.
    base = f"{chat_type}:{chat_id}"
    eid = f"{base}:thread:{thread_id}" if thread_id else base
    s = _blank_source("chat", eid)
    s["title"] = title or ""
    s["chat_type"] = chat_type
    s["chat_id"] = str(chat_id) if chat_id is not None else None
    s["thread_id"] = str(thread_id) if thread_id else None
    return s


def _note_source(
    note_type: Any,
    note_id: Any,
    title: Any = None,
    parent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    s = _blank_source("note", f"note:{note_type}:{note_id}")
    s["title"] = title or ""
    s["note_id"] = str(note_id) if note_id is not None else None
    s["note_type"] = note_type
    pc = parent_context or {}
    s["project_id"] = pc.get("project_id")
    s["task_id"] = pc.get("task_id")
    s["chat_type"] = pc.get("chat_type")
    s["chat_id"] = pc.get("chat_id")
    s["thread_id"] = pc.get("thread_id")
    return s


def _todo_source(item_id: Any, title: Any, local_date: Any) -> dict[str, Any]:
    # entity_id mirrors the chunker shape: `todo:YYYY-MM-DD:item:<id>`.
    # `related_entity_ids` points at the day-level grouping so a future
    # daily-summary chunker can co-link.
    s = _blank_source("todo", f"todo:{local_date}:item:{item_id}")
    s["title"] = title or ""
    s["related_entity_ids"] = [f"todo:{local_date}"]
    return s


def _milestone_source(
    milestone_id: Any, title: Any, project_id: Any, task_id: Any = None
) -> dict[str, Any]:
    # entity_id mirrors the milestone chunker (`milestone:<id>`). The
    # backing task_id + project_id let the frontend deep-link the chip
    # through the task view (App.tsx handleSpotlightSelect routes a
    # milestone the same as its backing task). task_id is None for legacy
    # milestones whose backing task hasn't been auto-created yet.
    s = _blank_source("milestone", f"milestone:{milestone_id}")
    s["title"] = title or ""
    s["project_id"] = str(project_id) if project_id is not None else None
    s["task_id"] = str(task_id) if task_id is not None else None
    return s


def _ui_sources_from_tool_result(call_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build UI source dicts from a non-search read tool's result.

    Returns [] for tools whose results don't map to a clickable entity
    (e.g. analytics aggregations without per-row ids, get_current_user,
    get_team_members — no user-detail view exists to link to).

    Sources are deduped upstream by (entity_type, entity_id), so emitting
    the same task from both `list_tasks` and `search_knowledge_base` in
    one run only produces a single chip.
    """
    if not isinstance(result, dict):
        return []

    if call_name in ("list_tasks", "get_stale_tasks"):
        tasks = result.get("tasks") or []
        task_sources = [
            _task_source(
                t.get("task_id"),
                t.get("title"),
                t.get("project_id"),
                display_id=t.get("display_id"),
            )
            for t in tasks
            if t.get("task_id")
        ]
        # Also emit one source per distinct project so inline
        # `[project:N]` citations the model writes (e.g.
        # "In Q2 Roadmap [project:18]: ...") resolve via the frontend
        # rewriter — without this, the bare token renders raw. Phase 4.2
        # citation-density ranking pushes uncited project chips down,
        # so unprompted-chip noise is bounded.
        seen_project_ids: set[Any] = set()
        project_sources: list[dict[str, Any]] = []
        for t in tasks:
            pid = t.get("project_id")
            if pid is None or pid in seen_project_ids:
                continue
            seen_project_ids.add(pid)
            project_sources.append(_project_source(pid, t.get("project_name")))
        return task_sources + project_sources

    if call_name == "fetch_task":
        tid = result.get("task_id")
        if not tid:
            return []
        return [
            _task_source(
                tid,
                result.get("title"),
                result.get("project_id"),
                display_id=result.get("display_id"),
            )
        ]

    if call_name == "create_task_plan":
        # Approved plan → chips for everything it created, so the final
        # answer's citations resolve and the user can click straight
        # into the new milestone/tasks. Capped like other multi-row chips.
        sources: list[dict[str, Any]] = []
        pid = result.get("project_id")
        ms = result.get("milestone") or {}
        if ms.get("milestone_id"):
            sources.append(
                _milestone_source(ms["milestone_id"], ms.get("title"), pid, ms.get("task_id"))
            )
        parent = result.get("parent_task") or {}
        if parent.get("task_id"):
            # Sub-task mode: chip for the anchor task the batch nested under.
            sources.append(
                _task_source(
                    parent["task_id"],
                    parent.get("title"),
                    pid,
                    display_id=parent.get("display_id"),
                )
            )
        for t in (result.get("tasks") or [])[:10]:
            if t.get("task_id"):
                sources.append(
                    _task_source(t["task_id"], t.get("title"), pid, display_id=t.get("display_id"))
                )
        if pid:
            sources.append(_project_source(pid, result.get("project_name")))
        return sources

    if call_name == "update_tasks_bulk":
        # Approved organize pass → chips for the touched tasks so the
        # answer's "I bumped X to Critical" citations resolve.
        return [
            _task_source(
                row.get("task_id"),
                row.get("title"),
                row.get("project_id"),
                display_id=row.get("display_id"),
            )
            for row in (result.get("updated") or [])[:10]
            if row.get("task_id")
        ]

    if call_name == "list_projects":
        return [
            _project_source(p.get("project_id"), p.get("project_name"))
            for p in (result.get("projects") or [])
            if p.get("project_id")
        ]

    if call_name == "get_project_summary":
        pid = result.get("project_id")
        if not pid:
            return []
        return [_project_source(pid, result.get("project_name"))]

    if call_name == "fetch_chat_thread":
        chat_type = result.get("chat_type")
        chat_id = result.get("chat_id")
        if not chat_type or not chat_id:
            return []
        return [_chat_source(chat_type, chat_id, result.get("thread_id"))]

    if call_name in ("fetch_note", "create_note", "update_note"):
        # The two write tools return the same note_id / note_type /
        # title / parent_context fields as fetch_note, so one branch
        # gives approved note writes a clickable chip too (emitted via
        # the resume path's write-result pass).
        nid = result.get("note_id")
        ntype = result.get("note_type")
        if not nid or not ntype:
            return []
        return [_note_source(ntype, nid, result.get("title"), result.get("parent_context"))]

    if call_name in ("list_milestones", "list_sprints"):
        # Milestones / sprints aren't a Spotlight entity_type today —
        # emit one project chip per distinct project so the user has at
        # least a deep-link surface to the right project.
        rows = result.get("milestones") if call_name == "list_milestones" else result.get(
            "sprints"
        )
        rows = rows or []
        seen: set[Any] = set()
        sources: list[dict[str, Any]] = []
        for r in rows:
            pid = r.get("project_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            sources.append(_project_source(pid, r.get("project_name")))
        return sources

    if call_name in ("get_milestone_summary", "get_sprint_summary"):
        pid = result.get("project_id")
        if not pid:
            return []
        return [_project_source(pid, result.get("project_name"))]

    if call_name == "get_team_task_summary":
        return [
            _project_source(p.get("project_id"), p.get("project_name"))
            for p in (result.get("per_project") or [])
            if p.get("project_id")
        ]

    if call_name == "get_task_blockers":
        # Emit the target task + each blocker/blocked task as separate
        # chips so the user can click into any of them. Dedup is by
        # (entity_type, entity_id) one level up, so emitting them all is
        # safe even when the graph self-references.
        sources: list[dict[str, Any]] = []
        target_tid = result.get("task_id")
        if target_tid is not None:
            sources.append(
                _task_source(
                    target_tid,
                    result.get("title"),
                    None,  # target's project_id isn't echoed — chip stays simple
                    display_id=result.get("display_id"),
                )
            )
        seen_projects: set[Any] = set()
        for direction in ("blocked_by", "blocking"):
            for ref in result.get(direction) or []:
                tid = ref.get("task_id")
                if tid is None:
                    continue
                sources.append(
                    _task_source(
                        tid,
                        ref.get("title"),
                        ref.get("project_id"),
                        display_id=ref.get("display_id"),
                    )
                )
                pid = ref.get("project_id")
                if pid is not None and pid not in seen_projects:
                    seen_projects.add(pid)
                    sources.append(_project_source(pid, ref.get("project_name")))
        return sources

    # --- Phase 18: me-scoped tools ---
    # Same chip-shape choices as the workspace-scoped counterparts so the
    # UI rewriter resolves citations identically whether the model came
    # in via `list_tasks` or `get_my_focus_tasks`.

    if call_name in ("get_my_focus_tasks", "get_my_schedule"):
        # Both return task rows (`tasks` / `tasks_due`). Same chip pattern
        # as `list_tasks`: one task chip per row + one project chip per
        # distinct project.
        key = "tasks" if call_name == "get_my_focus_tasks" else "tasks_due"
        rows = result.get(key) or []
        out: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for t in rows:
            tid = t.get("task_id")
            if tid is None:
                continue
            out.append(
                _task_source(
                    tid,
                    t.get("title"),
                    t.get("project_id"),
                    display_id=t.get("display_id"),
                )
            )
        for t in rows:
            pid = t.get("project_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out.append(_project_source(pid, t.get("project_name")))
        return out

    if call_name == "get_my_task_summary":
        return [
            _project_source(p.get("project_id"), p.get("project_name"))
            for p in (result.get("per_project") or [])
            if p.get("project_id")
        ]

    if call_name == "list_my_milestones":
        seen: set[Any] = set()
        out_proj: list[dict[str, Any]] = []
        for m in result.get("milestones") or []:
            pid = m.get("project_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out_proj.append(_project_source(pid, m.get("project_name")))
        return out_proj

    if call_name == "get_my_blockers":
        # Walk both halves; emit a task chip for each of my tasks and
        # for each edge endpoint, plus one project chip per distinct
        # project. Mirrors `get_task_blockers` shape choice.
        out_b: list[dict[str, Any]] = []
        seen_proj: set[Any] = set()
        for half_key, edge_key in (
            ("blocked_on_me", "blocked_by"),
            ("blocking_others", "blocking"),
        ):
            for row in result.get(half_key) or []:
                tid = row.get("task_id")
                if tid is not None:
                    out_b.append(
                        _task_source(
                            tid,
                            row.get("title"),
                            row.get("project_id"),
                            display_id=row.get("display_id"),
                        )
                    )
                pid = row.get("project_id")
                if pid is not None and pid not in seen_proj:
                    seen_proj.add(pid)
                    out_b.append(_project_source(pid, row.get("project_name")))
                for edge in row.get(edge_key) or []:
                    etid = edge.get("task_id")
                    if etid is None:
                        continue
                    out_b.append(
                        _task_source(
                            etid,
                            edge.get("title"),
                            edge.get("project_id"),
                            display_id=edge.get("display_id"),
                        )
                    )
                    epid = edge.get("project_id")
                    if epid is not None and epid not in seen_proj:
                        seen_proj.add(epid)
                        out_b.append(_project_source(epid, edge.get("project_name")))
        return out_b

    if call_name == "get_my_throughput":
        out_thr: list[dict[str, Any]] = []
        seen_thr: set[Any] = set()
        for t in result.get("recently_closed") or []:
            tid = t.get("task_id")
            if tid is None:
                continue
            out_thr.append(
                _task_source(
                    tid,
                    t.get("title"),
                    t.get("project_id"),
                    display_id=t.get("display_id"),
                )
            )
            pid = t.get("project_id")
            if pid is not None and pid not in seen_thr:
                seen_thr.add(pid)
                out_thr.append(_project_source(pid, t.get("project_name")))
        return out_thr

    if call_name == "list_my_mentions":
        # Each mention is addressed at a chat (thread or channel). The
        # chip lets the user click straight into that chat — same shape
        # the chunker emits for chat sources.
        out_m: list[dict[str, Any]] = []
        seen_m: set[tuple[Any, Any, Any]] = set()
        for row in result.get("mentions") or []:
            label = row.get("chat_type_label")
            cid = row.get("chat_id")
            tid_chat = row.get("thread_id") if row.get("is_thread") else None
            if not label or cid is None:
                continue
            key = (label, cid, tid_chat)
            if key in seen_m:
                continue
            seen_m.add(key)
            out_m.append(_chat_source(label, cid, tid_chat))
        return out_m

    # list_my_inbox: items don't map to a clickable Spotlight entity_type
    # today (no inbox-deep-link surface). Return [] so the agent has to
    # describe the items in prose rather than emit broken chips.

    # --- Todo tools ---
    if call_name == "list_today_todos":
        local_date = result.get("local_date")
        if not local_date:
            return []
        return [
            _todo_source(i.get("item_id"), i.get("title"), local_date)
            for i in result.get("items") or []
            if i.get("item_id")
        ]

    if call_name == "list_uncompleted_todos":
        return [
            _todo_source(i.get("item_id"), i.get("title"), i.get("local_date"))
            for i in result.get("items") or []
            if i.get("item_id") and i.get("local_date")
        ]

    if call_name in ("create_todo_item", "update_todo_item"):
        iid = result.get("item_id")
        # `update_todo_item` doesn't echo local_date; fall back to the
        # group-level prefix that's still resolvable on the frontend.
        ld = result.get("local_date") or ""
        if iid is None:
            return []
        return [_todo_source(iid, result.get("title"), ld)]

    return []


def reconstruct_sources_for_run(run) -> list[dict[str, Any]]:
    """Rebuild the same source list the live `/ask/` flow emitted for
    this run, replaying against persisted `AgentStep.result_json`.

    Walks the run's steps in insertion order and dispatches each one
    through the same per-tool source builders the live loop uses
    (`_ui_source_for_match` for `search_knowledge_base` matches,
    `_ui_sources_from_tool_result` for structured reads). Dedupes by
    `entity_id` so a task touched by both `list_tasks` and a follow-up
    `fetch_task` produces a single source row — matching the live
    `seen_sources_by_id` behavior in `_drive_loop`.

    Used by the History detail endpoint (so archived citation tokens
    resolve to clickable previews) AND by the `spotlight_answer`
    chunker (to derive each collected answer's provenance + ACL).
    The caller is responsible for prefetching `run.steps` if it wants
    to avoid an extra query.
    """
    seen_by_id: dict[str, dict[str, Any]] = {}
    for step in run.steps.all():
        if not step.tool_name or step.result_json is None:
            continue
        result = step.result_json
        if step.tool_name == "search_knowledge_base":
            new_sources = [_ui_source_for_match(m) for m in (result.get("matches") or [])]
        else:
            new_sources = _ui_sources_from_tool_result(step.tool_name, result)
        for s in new_sources:
            eid = s.get("entity_id")
            if eid and eid not in seen_by_id:
                seen_by_id[eid] = s
    return list(seen_by_id.values())


# --------------------------------------------------------------------------- #
# Phase 4.2 — source-chip ranking by citation density                         #
# --------------------------------------------------------------------------- #

# Matches BOTH citation forms the agent emits (§4.6 D5): the natural-prose
# link `[prose](type:id)` (id captured in group 1) and the bare `[type:id]`
# fallback (id in group 2). The link alternative is FIRST and consumes the
# whole `[label](id)` so a single-word label (`[spike](task:42)`) yields the
# id, not the label. Keep in sync with the frontend rewriter (citationUtils.ts
# CITATION_PATTERN / CITATION_LINK_PATTERN) and `_CITATION_RE` in
# evals/runner.py.
_INLINE_CITATION_RE = re.compile(r"\[[^\]]*\]\(([a-z][a-z0-9_:\-]+)\)|\[([a-z][a-z0-9_:\-]+)\]")


def _iter_cited_ids(text: str):
    """Yield the entity-id token from every citation (either form) in `text`."""
    for m in _INLINE_CITATION_RE.finditer(text or ""):
        yield (m.group(1) or m.group(2))


def _rank_sources_by_citation(
    answer_text: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-sort the source list so chips actually cited in the answer
    surface leftmost. Stable for sources with the same citation count
    (preserves original tool-emission order — which is already a
    reasonable secondary signal since the most-relevant tool usually
    fires first).

    Citation-id matching mirrors the frontend's `sourcesById` lookup
    (see Phase 0.3): citation tokens are always `<type>:<rest>`, but
    some entity types (chats) ship `entity_id` without the leading
    `<type>:` prefix. Normalise to the prefixed form for matching so
    `chat:dm:9:thread:4` in the answer text finds a chat source whose
    entity_id is `dm:9:thread:4`.
    """
    if not sources:
        return sources

    cited_tokens = {tok.lower() for tok in _iter_cited_ids(answer_text)}
    if not cited_tokens:
        return sources

    def _token_key(src: dict[str, Any]) -> str:
        etype = (src.get("entity_type") or "").lower()
        eid = (src.get("entity_id") or "").lower()
        if not eid:
            return ""
        return eid if eid.startswith(f"{etype}:") else f"{etype}:{eid}"

    def _citation_count(src: dict[str, Any]) -> int:
        key = _token_key(src)
        return 1 if key and key in cited_tokens else 0

    # Stable sort by (cited > uncited). Python's `sorted` is stable so
    # within-bucket order is preserved (= original tool-emission order).
    return sorted(sources, key=lambda s: -_citation_count(s))


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def run_agent(
    query: str,
    ctx: ToolContext,
    emit: Callable[[dict[str, Any]], None],
    *,
    run_id: UUID | None = None,
    prior_turns: list[tuple[str, str]] | None = None,
    prior_summary: str | None = None,
    disabled_tools: set[str] | None = None,
    system_extra: str | None = None,
    seed_sources: list[dict[str, Any]] | None = None,
    trace_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Drive the agent loop from a fresh user query.

    `prior_turns` is an ordered list of (user_query, assistant_answer)
    pairs from earlier turns in the same session (Phase 8). When
    present they are prepended to the messages list so the model can
    resolve references like "that task" or "the note you mentioned".
    Each answer is already truncated to ~400 chars by the view layer
    to keep the context budget bounded.

    `seed_sources` is an optional list of pre-built source dicts to
    register *before* the loop starts. Used by the note / thread Q&A
    branches: the agent has the summary in its system prompt and may
    answer without ever calling a tool, but it can still emit a
    `[note:...]` or `[chat:...]` citation for the entity the user
    opened the modal from. Pre-seeding the source lets the frontend
    citation rewriter resolve those tokens to a titled link instead
    of rendering the raw bracketed id.

    Returns:
        None on clean completion (text answer, error, or step cap).
        A pause descriptor when the loop hits a write tool:
            {
                "paused": True,
                "approval_token": UUID,
                "step": int,
                "tool_name": str,
                "arguments": dict,
            }
        The view layer reflects the pause back onto the `AgentRun` row.
    """
    # Pre-populate the live source map and ship the initial chips. The
    # event MUST be emitted before the first model call so even a zero-
    # tool answer still gets the seeded sources to the frontend.
    seeded_map: dict[tuple, dict[str, Any]] = {}
    if seed_sources:
        _apply_friendly_titles(seed_sources, ctx)
        _hydrate_task_display_ids(seed_sources)
        for src in seed_sources:
            key = (src.get("entity_type"), src.get("entity_id"))
            if not all(key) or key in seeded_map:
                continue
            seeded_map[key] = src
        if seeded_map:
            emit({"type": "sources", "sources": list(seeded_map.values())})

    messages: list[AgentMessage] = []
    # Phase 3.5 — rolling summary of earlier turns prepended as an
    # assistant "note to self" so the model can reference topics that
    # have fallen out of the verbatim prior_turns window. Cheap, opt-in
    # context recovery for long sessions. See `multi_turn.py`.
    if prior_summary:
        messages.append(
            AgentMessage(
                role="assistant",
                text=f"[Context recap from earlier in this conversation: {prior_summary}]",
            )
        )
    for prior_query, prior_answer in prior_turns or []:
        messages.append(_user_turn(prior_query))
        messages.append(AgentMessage(role="assistant", text=prior_answer))
    messages.append(_user_turn(query))

    # §4.5 — per-context tool subsetting. Drop peripheral families the
    # query shows no signal for, unioned with anything the caller already
    # disabled (e.g. the web-search toggle). Cuts the declared tool
    # surface; never touches core tools. Off by default.
    if settings.SEARCH_ENGINE.get("RAG_TOOL_SUBSETTING", False):
        excluded = _irrelevant_tool_families(query)
        if excluded:
            disabled_tools = (disabled_tools or set()) | excluded
            log.info(
                "Tool subsetting: dropped %d peripheral tool(s); %d declared",
                len(excluded),
                len(REGISTRY) - len(disabled_tools),
            )

    # Phase 3.2 — optional self-critique pass. Dispatched here so the
    # resume_agent path (write-tool approval flow) is NOT critiqued;
    # critique only makes sense on a complete, un-paused turn.
    def _inner(
        emit_fn: Callable[[dict[str, Any]], None],
        trace_fn: Callable[[str, dict[str, Any], dict[str, Any]], None] | None,
    ) -> dict[str, Any] | None:
        if settings.SEARCH_ENGINE.get("RAG_AGENT_SELF_CRITIQUE", False):
            return _drive_loop_with_critique(
                user_query=query,
                messages=messages,
                ctx=ctx,
                emit=emit_fn,
                run_id=run_id,
                starting_step=0,
                seen_sources_by_id=seeded_map,
                disabled_tools=disabled_tools,
                system_extra=system_extra,
                trace_hook=trace_fn,
                session_id=session_id,
            )
        return _drive_loop(
            messages=messages,
            ctx=ctx,
            emit=emit_fn,
            run_id=run_id,
            starting_step=0,
            seen_sources_by_id=seeded_map,
            disabled_tools=disabled_tools,
            system_extra=system_extra,
            trace_hook=trace_fn,
            session_id=session_id,
        )

    # §4.1 abstention gate — only on a fresh workspace query. With
    # `seed_sources` (thread/note Q&A) or `prior_turns` (multi-turn) the
    # answer can be grounded in context this gate can't see, so skip it
    # there to avoid false abstentions.
    gate_on = (
        settings.SEARCH_ENGINE.get("RAG_ABSTENTION_GATE", False)
        and not seed_sources
        and not prior_turns
    )
    if gate_on:
        return _run_with_abstention_gate(_inner, emit, trace_hook)
    return _inner(emit, trace_hook)


def resume_agent(
    run: AgentRun,
    decision: str,
    ctx: ToolContext,
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Resume a paused agent run after the user has approved or rejected.

    Reconstructs the conversation up to the pending tool call from
    `AgentStep` rows, executes (approve) or synthesizes a rejection
    (reject) for that one tool, then continues the loop. Returns
    `None` on completion or another pause descriptor if the resumed
    run hits a second write tool.
    """
    if decision not in (DECISION_APPROVE, DECISION_REJECT):
        emit(
            {
                "type": "error",
                "message": f"Invalid decision {decision!r} (expected 'approve' or 'reject').",
            }
        )
        return None

    messages, pending_step = _rebuild_messages(run)
    if pending_step is None:
        emit(
            {
                "type": "error",
                "message": "No pending tool call found on this run.",
            }
        )
        return None

    step_index = pending_step.step_index
    call_name = pending_step.tool_name
    call_args = dict(pending_step.arguments_json or {})
    function_call = FunctionCall(
        name=call_name,
        args=call_args,
        thought_signature=_coerce_signature(pending_step.thought_signature),
    )
    # Chips created by an approved write (filled in the approve branch);
    # doubles as the continued loop's sources dedup seed.
    resumed_sources: dict[tuple[Any, Any], dict[str, Any]] = {}

    # Emit the start event the original run skipped. Same step index so
    # the frontend can correlate the approve/reject card with the row
    # that's now actually executing.
    emit(
        {
            "type": "tool_call_start",
            "step": step_index,
            "tool_name": call_name,
            "arguments": _friendly_arguments(call_args, call_name),
        }
    )

    if decision == DECISION_REJECT:
        err = "User rejected this action."
        emit(
            {
                "type": "tool_call_error",
                "step": step_index,
                "tool_name": call_name,
                "error": err,
            }
        )
        try:
            pending_step.error = "user_rejected"
            pending_step.summary = ""
            pending_step.save(update_fields=["error", "summary"])
        except Exception:  # noqa: BLE001
            log.exception("Failed to update pending step %s on reject", pending_step.step_id)
        messages.append(_assistant_function_call_turn(function_call))
        messages.append(_function_response_turn(call_name, {"error": "user_rejected"}))
    else:
        # APPROVE — actually run the tool now.
        tool = REGISTRY.get(call_name)
        if tool is None:
            err = f"Unknown tool: {call_name}"
            emit(
                {
                    "type": "tool_call_error",
                    "step": step_index,
                    "tool_name": call_name,
                    "error": err,
                }
            )
            try:
                pending_step.error = err
                pending_step.summary = ""
                pending_step.save(update_fields=["error", "summary"])
            except Exception:  # noqa: BLE001
                log.exception(
                    "Failed to update pending step %s on unknown tool", pending_step.step_id
                )
            messages.append(_assistant_function_call_turn(function_call))
            messages.append(_function_response_turn(call_name, {"error": err}))
        else:
            try:
                result = tool.run(call_args, ctx)
            except ToolError as e:
                emit(
                    {
                        "type": "tool_call_error",
                        "step": step_index,
                        "tool_name": call_name,
                        "error": str(e),
                    }
                )
                try:
                    pending_step.error = str(e)
                    pending_step.summary = ""
                    pending_step.save(update_fields=["error", "summary"])
                except Exception:  # noqa: BLE001
                    log.exception(
                        "Failed to update pending step %s after ToolError", pending_step.step_id
                    )
                messages.append(_assistant_function_call_turn(function_call))
                messages.append(_function_response_turn(call_name, {"error": str(e)}))
            except Exception as e:  # noqa: BLE001
                log.exception("Tool %s crashed on args %r", call_name, call_args)
                err = f"Internal error in tool '{call_name}'."
                emit(
                    {
                        "type": "tool_call_error",
                        "step": step_index,
                        "tool_name": call_name,
                        "error": err,
                    }
                )
                try:
                    pending_step.error = err
                    pending_step.summary = ""
                    pending_step.save(update_fields=["error", "summary"])
                except Exception:  # noqa: BLE001
                    log.exception(
                        "Failed to update pending step %s after exception", pending_step.step_id
                    )
                messages.append(_assistant_function_call_turn(function_call))
                messages.append(_function_response_turn(call_name, {"error": err}))
            else:
                summary = result.pop("__summary__", "ok")
                result_event = {
                    "type": "tool_call_result",
                    "step": step_index,
                    "tool_name": call_name,
                    "summary": summary,
                }
                # Approved note writes carry a compact `note` ref so the
                # frontend can refresh caches and (for body updates) push
                # the new blocks into the live Yjs doc. Additive — older
                # frontends destructure known fields and ignore this.
                if call_name in ("create_note", "update_note"):
                    note_ref = {
                        k: result.get(k)
                        for k in ("note_id", "note_type", "title", "changed_fields")
                        if result.get(k) is not None
                    }
                    if note_ref.get("note_id"):
                        result_event["note"] = note_ref
                emit(result_event)
                # C3: an APPROVED write just changed the workspace, so
                # every cached read this session made before it is now
                # suspect — drop them all (generation bump, O(1)).
                if run.session_id:
                    tool_cache.invalidate_session(str(run.session_id))
                try:
                    pending_step.summary = summary
                    pending_step.result_json = result
                    pending_step.save(update_fields=["summary", "result_json"])
                except Exception:  # noqa: BLE001
                    log.exception(
                        "Failed to update pending step %s after approve",
                        pending_step.step_id,
                    )
                # Chips for entities the approved write CREATED (e.g. a
                # create_task_plan's milestone + tasks) so the final
                # answer's citations resolve to clickable previews.
                # Mirrors the seeded-sources pattern in run_agent: emit
                # once here, then seed the continued loop's dedup map so
                # later read-tools don't re-add (and cumulative `sources`
                # events keep including them). Tools without a
                # _ui_sources_from_tool_result branch return [] — no
                # behavior change for the other write tools.
                for src in _ui_sources_from_tool_result(call_name, result):
                    key = (src.get("entity_type"), src.get("entity_id"))
                    if all(key) and key not in resumed_sources:
                        resumed_sources[key] = src
                if resumed_sources:
                    hydrated = _hydrate_task_display_ids(
                        _apply_friendly_titles(list(resumed_sources.values()), ctx)
                    )
                    emit({"type": "sources", "sources": hydrated})
                messages.append(_assistant_function_call_turn(function_call))
                messages.append(_function_response_turn(call_name, result))

    # Continue the loop from the next step. The original run wrote
    # steps 0..step_index inclusive, so we resume at step_index + 1.
    # `resumed_sources` seeds the dedup map with chips the approved
    # write emitted above; pre-approval `sources` events were already
    # sent in the original stream and are not re-seeded.
    return _drive_loop(
        messages=messages,
        ctx=ctx,
        emit=emit,
        run_id=run.run_id,
        starting_step=step_index + 1,
        seen_sources_by_id=resumed_sources,
        session_id=str(run.session_id) if run.session_id else None,
    )


# --------------------------------------------------------------------------- #
# Shared loop body                                                            #
# --------------------------------------------------------------------------- #


def _run_tool_guarded(tool: Any, call_name: str, call_args: dict[str, Any], ctx: ToolContext) -> tuple[str, Any]:
    """Execute one read-only tool, never raising.

    Returns a tagged outcome consumed by the loop's apply phase:
      ("ok", result_dict) | ("tool_error", message) | ("crash", message)
    The tag split preserves today's two error flavors exactly: a
    ToolError's message is user-facing (ACL denials, bad args), a crash
    gets the generic internal-error string + a server-side traceback.
    Shared by the serial path and the E1 parallel executor, so error
    semantics can't diverge between them.
    """
    try:
        return ("ok", tool.run(call_args, ctx))
    except ToolError as e:
        return ("tool_error", str(e))
    except Exception:  # noqa: BLE001
        log.exception("Tool %s crashed on args %r", call_name, call_args)
        return ("crash", f"Internal error in tool '{call_name}'.")


def _execute_batch_parallel(
    calls: list[FunctionCall], ctx: ToolContext
) -> list[tuple[str, Any]]:
    """E1 — run a batch of read-only tool calls concurrently.

    Returns outcomes IN CALL ORDER (pool.map preserves it), so the
    caller's emit/persist/messages sequence stays byte-deterministic
    regardless of completion order — `AgentStep` rows and the message
    transcript must not depend on thread scheduling (resume rebuilds
    from them). Wall-clock is bounded by the slowest call either way.

    Each worker closes its Django DB connections in `finally`:
    short-lived executor threads would otherwise leak a connection per
    call (`close_old_connections` is for long-lived threads; these die
    with the batch). All DB WRITES (_persist_step) happen on the
    controller thread after the join — no write concurrency here.
    """
    max_workers = min(
        len(calls), int(settings.SEARCH_ENGINE.get("RAG_PARALLEL_TOOLS_MAX_WORKERS", 4))
    )

    def _task(call: FunctionCall) -> tuple[str, Any]:
        try:
            return _run_tool_guarded(REGISTRY.get(call.name), call.name, dict(call.args), ctx)
        finally:
            try:
                connections.close_all()
            except Exception:  # noqa: BLE001 — cleanup must not eat the outcome
                log.exception("close_all failed in parallel tool worker")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_task, calls))
def _resolve_planning_override() -> str | None:
    """B3 provider tier split (SPOTLIGHT_FUTURE_ARCHITECTURE.md §3).

    Returns the model id to run the loop's PLANNING steps on (passed as
    `model_override`, which beats the user's `LlmChoice`), or None when
    the split is inactive and every step runs on one model exactly as
    before. The final SYNTHESIS step always uses the user's model — see
    the discard-and-rerun logic in `_drive_loop`.

    Skips (returning None) when:
      * `RAG_PLANNING_MODEL` is empty — feature off, zero code-path change;
      * the planning model IS the effective synthesis model (user picked
        the fast model, or the server default already is it) — the split
        would buy nothing but the buffering machinery;
      * the planning model's provider doesn't match the active provider.
        This guard is PREVENTIVE, unlike the reranker's try/except
        fallback: a mid-loop model error kills the run, so we must never
        hand `ClaudeClient` a `gemini-*` id (or vice versa).
    """
    planning = (settings.SEARCH_ENGINE.get("RAG_PLANNING_MODEL") or "").strip()
    if not planning:
        return None
    choice = get_llm_choice() or _server_default_choice()
    synthesis_model = (choice.model or "").strip()
    if not synthesis_model or planning == synthesis_model:
        return None
    provider_prefix = "claude" if choice.provider == "claude" else "gemini"
    if not planning.startswith(provider_prefix):
        log.warning(
            "RAG_PLANNING_MODEL=%r does not match the active provider %r; "
            "planning split disabled for this run",
            planning,
            choice.provider,
        )
        return None
    return planning


def _collect_step(
    client: Any,
    *,
    messages: list[AgentMessage],
    tools: list[ToolDeclaration],
    system_instruction: str,
    emit: Callable[[dict[str, Any]], None],
    model_override: str | None = None,
    emit_deltas: bool = True,
) -> tuple[list[str], list[FunctionCall]]:
    """Run ONE model turn and collect `(text_parts, function_calls)`.

    `emit_deltas=False` buffers text instead of streaming it — the B3
    planning pass uses this because a text-only planning response is a
    DRAFT that will be discarded and re-generated on the user's model;
    streaming it would show the user an answer that then gets replaced.
    Exceptions propagate to the caller (the loop owns error handling).
    """
    text_parts: list[str] = []
    calls: list[FunctionCall] = []
    stream = client.generate_step(
        messages=messages,
        tools=tools,
        system_instruction=system_instruction,
        model_override=model_override,
    )
    for text_chunk, function_call in stream:
        if function_call is not None:
            calls.append(function_call)
        elif text_chunk:
            text_parts.append(text_chunk)
            if emit_deltas:
                emit({"type": "answer_delta", "text": text_chunk})
    return text_parts, calls


def _drive_loop(
    *,
    messages: list[AgentMessage],
    ctx: ToolContext,
    emit: Callable[[dict[str, Any]], None],
    run_id: UUID | None,
    starting_step: int,
    seen_sources_by_id: dict[tuple, dict[str, Any]],
    disabled_tools: set[str] | None = None,
    system_extra: str | None = None,
    trace_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    max_steps: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """The core agent loop, shared by `run_agent` and `resume_agent`.

    Returns `None` on completion, or a pause descriptor on hitting a
    write tool. See `run_agent` for the descriptor shape.

    `system_extra`, when set, is appended to the canonical
    `AGENT_SYSTEM_PROMPT` before being passed to the model. The thread
    Q&A flow (AgentAskView's thread_context branch) uses this to inject
    the thread summary + a "stay scoped to this thread" directive.

    `max_steps`, when set, overrides the `AGENT_MAX_STEPS` ceiling for
    this call. The critique-with-retrieval continuation passes a tight
    budget (`starting_step + RAG_CRITIQUE_MAX_STEPS`) so the critique can
    fire at most one more retrieval before answering.

    `session_id`, when set, keys the C3 session tool-result cache — a
    follow-up turn re-calling the same read-only tool with identical
    args reuses the stored result. None (evals, tests, sessionless
    callers) bypasses the cache entirely.
    """
    if max_steps is None:
        max_steps = int(settings.SEARCH_ENGINE.get("AGENT_MAX_STEPS", 5))
    client = get_model_client()
    # B3 — planning-model split. When active, loop steps run on the fast
    # planning model and only the final synthesis is written by the
    # user's model, via discard-and-rerun below. None = single-model
    # behavior, byte-identical to the pre-B3 loop.
    planning_model = _resolve_planning_override()
    tools = _build_tool_declarations(disabled_tools)
    system_instruction = AGENT_SYSTEM_PROMPT
    if system_extra:
        system_instruction = f"{AGENT_SYSTEM_PROMPT}\n\n{system_extra}"

    for step in range(starting_step, max_steps):
        try:
            if planning_model is None:
                accumulated_text_parts, accumulated_function_calls = _collect_step(
                    client,
                    messages=messages,
                    tools=tools,
                    system_instruction=system_instruction,
                    emit=emit,
                )
            else:
                # Planning pass on the fast model, deltas BUFFERED — a
                # step is only known to be planning vs synthesis after
                # the model responds, and a synthesis draft from the
                # fast model must never reach the user.
                accumulated_text_parts, accumulated_function_calls = _collect_step(
                    client,
                    messages=messages,
                    tools=tools,
                    system_instruction=system_instruction,
                    emit=emit,
                    model_override=planning_model,
                    emit_deltas=False,
                )
                if accumulated_function_calls:
                    # Planning step confirmed — flush the buffered
                    # thinking-out-loud text now (same events, batched).
                    for chunk in accumulated_text_parts:
                        emit({"type": "answer_delta", "text": chunk})
                else:
                    # The fast model judged the loop ready to answer.
                    # DISCARD its draft and re-run this one step with no
                    # override (= the user's model), streaming live. Net
                    # cost vs single-model: the same one smart call plus
                    # N cheap planning calls and one wasted draft; net
                    # win: every planning round-trip at fast-model
                    # latency. If the smart model instead decides to dig
                    # further (it's the better judge), its calls flow
                    # into normal tool execution below.
                    accumulated_text_parts, accumulated_function_calls = _collect_step(
                        client,
                        messages=messages,
                        tools=tools,
                        system_instruction=system_instruction,
                        emit=emit,
                    )
        except Exception as e:  # noqa: BLE001 — surface as stream error
            log.exception("Agent step %d LLM call failed", step)
            emit({"type": "error", "message": f"LLM call failed: {e}"})
            _persist_step(run_id, step_index=step, error=f"LLM call failed: {e}")
            return None

        any_text_emitted = bool(accumulated_text_parts)

        if any_text_emitted:
            _persist_step(
                run_id,
                step_index=step,
                answer_text="".join(accumulated_text_parts),
            )

        if not accumulated_function_calls:
            if not any_text_emitted:
                emit(
                    {
                        "type": "error",
                        "message": "Model returned an empty response.",
                    }
                )
                _persist_step(run_id, step_index=step, error="empty_response")
                return None

            final_answer = "".join(accumulated_text_parts)

            # Post-process: resolve any `[type:id]` tokens in the final
            # answer that aren't already in the source registry. Common
            # causes: agent cited an entity carried over from a prior
            # turn, mentioned in a pre-injected summary, or otherwise
            # not retrieved via a tool this turn. Lookups are ACL-gated;
            # silent failure (raw token) is preferable to leaking titles.
            late_sources = resolve_unresolved_citations(
                answer=final_answer,
                seen_keys=set(seen_sources_by_id.keys()),
                team_id=ctx.team_id,
                user_id=ctx.user_id,
                build_task_source=lambda task_id, title, project_id: _task_source(
                    task_id, title, project_id
                ),
                build_project_source=lambda project_id, project_name: _project_source(
                    project_id, project_name
                ),
                build_chat_source=lambda chat_type, chat_id, thread_id: _chat_source(
                    chat_type, chat_id, thread_id
                ),
                build_note_source=lambda note_type, note_id, title, parent_context: _note_source(
                    note_type, note_id, title, parent_context
                ),
                build_todo_source=lambda item_id, title, local_date: _todo_source(
                    item_id, title, local_date
                ),
                build_milestone_source=lambda milestone_id, title, project_id, task_id: (
                    _milestone_source(milestone_id, title, project_id, task_id)
                ),
            )
            if late_sources:
                _apply_friendly_titles(late_sources, ctx)
                _hydrate_task_display_ids(late_sources)
                for src in late_sources:
                    key = (src.get("entity_type"), src.get("entity_id"))
                    if not all(key) or key in seen_sources_by_id:
                        continue
                    seen_sources_by_id[key] = src

            # Phase 4.2 — re-emit the sources list re-sorted by citation
            # density before `done`. Frontend already handles `sources`
            # events by replacing wholesale, so the final emit overrides
            # the in-flight tool-emission order with the more-relevant
            # citation-first order. Gated on a flag (default True) so
            # operators can flip it off if the chip-reshuffle UX bites.
            if (
                settings.SEARCH_ENGINE.get("RAG_RANK_SOURCES_BY_CITATION", True)
                and seen_sources_by_id
            ):
                ranked = _rank_sources_by_citation(final_answer, list(seen_sources_by_id.values()))
                emit({"type": "sources", "sources": ranked})
            elif late_sources:
                # Rank flag off but we added sources after the last tool
                # emit — ship the updated list so the frontend rewriter
                # has them.
                emit({"type": "sources", "sources": list(seen_sources_by_id.values())})
            emit({"type": "done"})
            return None

        # ---- C3: resolve session-cache hits before any execution ----
        # A follow-up turn re-calling the same read-only tool with the
        # same args inside the session TTL gets the stored result — no
        # tool round-trip. Hits are injected as pre-resolved "ok"
        # outcomes (summary re-attached so the shared pop below works);
        # write/unknown tools are never consulted.
        precomputed: dict[int, tuple[str, Any]] = {}
        cached_indices: set[int] = set()
        if session_id and tool_cache.enabled():
            for i, c in enumerate(accumulated_function_calls):
                t = REGISTRY.get(c.name)
                if t is None or getattr(t, "requires_approval", False):
                    continue
                hit = tool_cache.get_cached(session_id, c.name, dict(c.args))
                if hit is not None:
                    precomputed[i] = ("ok", {**hit["result"], "__summary__": hit["summary"]})
                    cached_indices.add(i)

        # ---- E1: parallel dispatch for read-only batches (flag-gated) ----
        # Only when the WHOLE batch is known, read-only tools and there
        # is actual parallelism to win among the cache MISSES (>1). Any
        # unknown tool or `requires_approval` write keeps the serial
        # path byte-for-byte — the approval pause returns mid-batch and
        # drops the remaining calls, and "parallel + mid-stream pause"
        # is incoherent. All `tool_call_start` events go out first in
        # call order; outcomes come back in call order too (see
        # _execute_batch_parallel), so the emit/persist/messages
        # sequence below is unchanged.
        starts_pre_emitted = False
        miss_indices = [
            i for i in range(len(accumulated_function_calls)) if i not in precomputed
        ]
        if (
            settings.SEARCH_ENGINE.get("RAG_PARALLEL_TOOLS", False)
            and len(miss_indices) > 1
            and all(
                REGISTRY.get(c.name) is not None
                and not getattr(REGISTRY.get(c.name), "requires_approval", False)
                for c in accumulated_function_calls
            )
        ):
            for call in accumulated_function_calls:
                emit(
                    {
                        "type": "tool_call_start",
                        "step": step,
                        "tool_name": call.name,
                        "arguments": _friendly_arguments(dict(call.args), call.name),
                    }
                )
            starts_pre_emitted = True
            miss_calls = [accumulated_function_calls[i] for i in miss_indices]
            for i, outcome in zip(miss_indices, _execute_batch_parallel(miss_calls, ctx)):
                precomputed[i] = outcome

        for call_idx, call in enumerate(accumulated_function_calls):
            call_args = dict(call.args)
            call_name = call.name

            tool = REGISTRY.get(call_name)
            if tool is None:
                emit(
                    {
                        "type": "tool_call_start",
                        "step": step,
                        "tool_name": call_name,
                        "arguments": _friendly_arguments(call_args, call_name),
                    }
                )
                err = f"Unknown tool: {call_name}"
                emit(
                    {
                        "type": "tool_call_error",
                        "step": step,
                        "tool_name": call_name,
                        "error": err,
                    }
                )
                _persist_step(
                    run_id,
                    step_index=step,
                    tool_name=call_name,
                    arguments_json=call_args,
                    thought_signature=call.thought_signature,
                    error=err,
                )
                messages.append(_assistant_function_call_turn(call))
                messages.append(_function_response_turn(call_name, {"error": err}))
                continue

            # ---- Phase 7: write tools pause the loop ----
            if getattr(tool, "requires_approval", False):
                approval_token = uuid.uuid4()
                _persist_step(
                    run_id,
                    step_index=step,
                    tool_name=call_name,
                    arguments_json=call_args,
                    thought_signature=call.thought_signature,
                    summary=PENDING_APPROVAL_MARKER,
                )
                # `run_id` is included so the frontend has everything it
                # needs to POST `/decide/`. When `run_id` is None (eval
                # / test paths) we omit the field rather than serialize
                # a `null` that the wire schema doesn't expect.
                event: dict[str, Any] = {
                    "type": "tool_call_pending_approval",
                    "step": step,
                    "tool_name": call_name,
                    "arguments": _friendly_arguments(call_args, call_name),
                    "approval_token": str(approval_token),
                }
                if run_id is not None:
                    event["run_id"] = str(run_id)
                emit(event)
                return {
                    "paused": True,
                    "approval_token": approval_token,
                    "step": step,
                    "tool_name": call_name,
                    "arguments": call_args,
                }

            # ---- Read-only tool ----
            # Serial path: emit start + resolve inline (cache hit or
            # run). Parallel path: starts were already emitted (call
            # order) and the outcome sits in `precomputed` — everything
            # from here down is identical for both, so E1/C3 cannot
            # change events, AgentStep rows, or the message transcript,
            # only whether/when `tool.run` executed.
            if not starts_pre_emitted:
                emit(
                    {
                        "type": "tool_call_start",
                        "step": step,
                        "tool_name": call_name,
                        "arguments": _friendly_arguments(call_args, call_name),
                    }
                )
            if call_idx in precomputed:
                kind, payload = precomputed[call_idx]
            else:
                kind, payload = _run_tool_guarded(tool, call_name, call_args, ctx)
            from_cache = call_idx in cached_indices

            if kind != "ok":
                err = str(payload)
                emit(
                    {
                        "type": "tool_call_error",
                        "step": step,
                        "tool_name": call_name,
                        "error": err,
                    }
                )
                _persist_step(
                    run_id,
                    step_index=step,
                    tool_name=call_name,
                    arguments_json=call_args,
                    thought_signature=call.thought_signature,
                    error=err,
                )
                if trace_hook is not None:
                    try:
                        trace_hook(call_name, call_args, {"error": err})
                    except Exception:  # noqa: BLE001
                        log.exception("trace_hook failed for tool %s (error path)", call_name)
                messages.append(_assistant_function_call_turn(call))
                messages.append(_function_response_turn(call_name, {"error": err}))
                continue

            result = payload
            summary = result.pop("__summary__", "ok")
            result_event = {
                "type": "tool_call_result",
                "step": step,
                "tool_name": call_name,
                "summary": summary,
            }
            if from_cache:
                # Observability marker only — the frontend destructures
                # step/tool_name/summary and ignores unknown fields.
                result_event["cached"] = True
            emit(result_event)
            # C3: cache the fresh result for follow-up turns in this
            # session. Cached hits are NOT re-stored (their TTL should
            # date from the original execution, not the last reuse).
            if not from_cache:
                tool_cache.store(session_id, call_name, call_args, summary, result)
            _persist_step(
                run_id,
                step_index=step,
                tool_name=call_name,
                arguments_json=call_args,
                thought_signature=call.thought_signature,
                summary=summary,
                result_json=result,
            )
            if trace_hook is not None:
                try:
                    trace_hook(call_name, call_args, result)
                except Exception:  # noqa: BLE001 — trace hook must never break the loop
                    log.exception("trace_hook failed for tool %s", call_name)

            # Collect citation chips from this tool's result. Search produces
            # them via _ui_source_for_match (one per match); structured read
            # tools produce them via _ui_sources_from_tool_result. Both feed
            # the same dedup map so a task surfaced by both list_tasks and
            # search_knowledge_base in one run is still a single chip.
            new_sources: list[dict[str, Any]] = []
            if call_name == "search_knowledge_base":
                new_sources = [_ui_source_for_match(m) for m in result.get("matches", [])]
            else:
                new_sources = _ui_sources_from_tool_result(call_name, result)

            # Swap viewer-agnostic placeholders ("DM 9") for friendly
            # titles (partner / group / project name) before chips ship.
            _apply_friendly_titles(new_sources, ctx)
            # Backfill PRJ-123 display ids for search-result task sources
            # (the index stores raw task_id only).
            _hydrate_task_display_ids(new_sources)

            added = False
            for src in new_sources:
                key = (src.get("entity_type"), src.get("entity_id"))
                if not all(key):
                    continue
                existing = seen_sources_by_id.get(key)
                if existing is None:
                    seen_sources_by_id[key] = src
                    added = True
                    continue
                # Same entity surfaced by two tools (e.g. fetch_chat_thread
                # then search_knowledge_base). First-writer-wins on the chip
                # itself, but upgrade the chat deep-link target if a later
                # source pinned the exact matched message/thread the first
                # one lacked — otherwise a message_id-less chip (fetch_*)
                # would suppress a message_id-bearing one (search) and the
                # click could only land at the chat top.
                if src.get("message_id") and not existing.get("message_id"):
                    existing["message_id"] = src.get("message_id")
                    if src.get("thread_id") and not existing.get("thread_id"):
                        existing["thread_id"] = src.get("thread_id")
                    added = True

            if added:
                emit(
                    {
                        "type": "sources",
                        "sources": list(seen_sources_by_id.values()),
                    }
                )

            messages.append(_assistant_function_call_turn(call))
            messages.append(_function_response_turn(call_name, result))

    # Step cap.
    emit(
        {
            "type": "error",
            "message": f"Agent did not reach a final answer in {max_steps} steps.",
        }
    )
    _persist_step(run_id, step_index=max_steps, error="step_cap_reached")
    return None


# --------------------------------------------------------------------------- #
# Phase 3.2 — self-critique reflection wrapper                                #
# --------------------------------------------------------------------------- #


def _drive_loop_with_critique(
    *,
    user_query: str,
    messages: list[AgentMessage],
    ctx: ToolContext,
    emit: Callable[[dict[str, Any]], None],
    run_id: UUID | None,
    starting_step: int,
    seen_sources_by_id: dict[tuple, dict[str, Any]],
    disabled_tools: set[str] | None = None,
    system_extra: str | None = None,
    trace_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Run `_drive_loop` with captured events, then optionally rewrite
    the draft answer via a single self-critique LLM call.

    Wrapper design (intentionally NOT inside `_drive_loop`): the inner
    loop is untouched and remains the canonical control path. The
    wrapper buffers events, runs a critique pass, then replays events
    to the real `emit` with the draft answer possibly swapped for a
    revised version.

    Precision-tightening only — the critique cannot fire more tool
    calls in this MVP. If a recall gap turns out to be the bottleneck
    on a future suite, extend the critique prompt to allow emitting
    a query the loop then executes.

    Tradeoff: TTFT becomes "end of loop + critique" because all
    answer_delta events are buffered. Acceptable for an experimental
    flag (off by default). Production rollout should weigh streaming
    vs. precision wins.

    Pause path (write-tool approval) is passed through unchanged — the
    `_drive_loop` returns a pause descriptor and the wrapper flushes
    captured events as-is. Critique never fires on a paused run.
    """
    captured_events: list[dict[str, Any]] = []
    captured_tool_results: list[dict[str, Any]] = []

    def _capture_emit(event: dict[str, Any]) -> None:
        captured_events.append(event)

    def _capture_trace(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured_tool_results.append({"tool_name": name, "arguments": args, "result": result})
        # Also forward to the caller's trace_hook if any (e.g. the eval runner).
        if trace_hook is not None:
            try:
                trace_hook(name, args, result)
            except Exception:  # noqa: BLE001
                log.exception("Outer trace_hook failed inside critique wrapper for %s", name)

    pause_descriptor = _drive_loop(
        messages=messages,
        ctx=ctx,
        emit=_capture_emit,
        run_id=run_id,
        starting_step=starting_step,
        seen_sources_by_id=seen_sources_by_id,
        disabled_tools=disabled_tools,
        system_extra=system_extra,
        trace_hook=_capture_trace,
        session_id=session_id,
    )

    if pause_descriptor is not None:
        # Loop paused on a write tool; do not critique. Flush as captured.
        for e in captured_events:
            emit(e)
        return pause_descriptor

    draft_answer = "".join(
        (e.get("text") or "") for e in captured_events if e.get("type") == "answer_delta"
    )
    if not draft_answer.strip():
        # No final answer to critique (e.g. step-cap, fatal error).
        for e in captured_events:
            emit(e)
        return None

    # Critique-with-retrieval path (RAG_CRITIQUE_RETRIEVAL). Runs a short,
    # read-only continuation of the loop so the critique can re-retrieve to
    # fix a completeness gap, not just rewrite text. Falls through to the
    # precision-only path below when the sub-flag is off, leaving that
    # measured behavior untouched.
    if bool(settings.SEARCH_ENGINE.get("RAG_CRITIQUE_RETRIEVAL", False)):
        for e in _critique_with_retrieval(
            loop1_events=captured_events,
            draft=draft_answer,
            messages=messages,
            ctx=ctx,
            run_id=run_id,
            starting_step=starting_step,
            seen_sources_by_id=seen_sources_by_id,
            disabled_tools=disabled_tools,
            system_extra=system_extra,
            trace_hook=trace_hook,
            session_id=session_id,
        ):
            emit(e)
        return None

    try:
        revised = _run_self_critique(
            user_query=user_query,
            tool_results=captured_tool_results,
            draft=draft_answer,
        )
    except Exception:  # noqa: BLE001 — never break the loop on critique failure
        log.exception("Self-critique LLM call failed; emitting draft unchanged")
        for e in captured_events:
            emit(e)
        return None

    if revised is None or _critique_says_keep(revised):
        # KEEP path — flush as captured.
        for e in captured_events:
            emit(e)
        return None

    # Revise path — replay everything except the draft answer_delta
    # events, then emit the revised answer once (just before `done`).
    revised_emitted = False
    for e in captured_events:
        etype = e.get("type")
        if etype == "answer_delta":
            # Drop the draft text.
            continue
        if etype == "done" and not revised_emitted:
            emit({"type": "answer_delta", "text": revised})
            revised_emitted = True
        emit(e)
    # Defensive: if there was no `done` event in the capture (shouldn't
    # happen for a clean termination) but we have a revision, surface it.
    if not revised_emitted:
        emit({"type": "answer_delta", "text": revised})
        emit({"type": "done"})
    return None


def _critique_says_keep(text: str) -> bool:
    """Recognise the literal KEEP signal. Anything else is a revision.

    Strict: only `"KEEP"` (case-insensitive) plus optional surrounding
    whitespace counts. If the model writes "KEEP, but actually …" or
    "Looks good — KEEP", treat it as a revision so we don't accidentally
    suppress a corrective rewrite.
    """
    return text.strip().upper() == "KEEP"


def _run_self_critique(
    *,
    user_query: str,
    tool_results: list[dict[str, Any]],
    draft: str,
) -> str | None:
    """Run one self-critique LLM call. Returns the model's text response
    (which may be the literal `KEEP` or a revised final answer).
    """
    prompt = AGENT_SELF_CRITIQUE_PROMPT_TEMPLATE.format(
        user_query=user_query,
        tool_summary=_format_tool_results_for_critique(tool_results),
        draft=draft,
    )
    client = get_model_client()
    chunks: list[str] = []
    for text, _fcall in client.generate_step(
        messages=[AgentMessage(role="user", text=prompt)],
        tools=[],
        system_instruction=AGENT_SELF_CRITIQUE_SYSTEM,
    ):
        if text:
            chunks.append(text)
    out = "".join(chunks).strip()
    return out or None


def _critique_with_retrieval(
    *,
    loop1_events: list[dict[str, Any]],
    draft: str,
    messages: list[AgentMessage],
    ctx: ToolContext,
    run_id: UUID | None,
    starting_step: int,
    seen_sources_by_id: dict[tuple, dict[str, Any]],
    disabled_tools: set[str] | None,
    system_extra: str | None,
    trace_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieval-capable critique: re-enter `_drive_loop` for a short,
    read-only continuation so the model can fix a *completeness* gap with
    one more retrieval, then merge the result over the draft.

    Returns the event list the caller should emit. The draft is preserved
    verbatim unless the continuation actually retrieved AND produced an
    answer — see `_merge_critique_events`.

    Reuses the full loop (tool dispatch, source emission, citation
    resolution) rather than re-implementing them: the new sources surface
    as chips and the revised answer's `[type:id]` tokens resolve exactly
    as in a normal turn, because `seen_sources_by_id` is shared.
    """
    # Continuation conversation: show the model its own draft, then the
    # completeness directive. `messages` already carries loop 1's tool
    # turns (mutated in place by `_drive_loop`).
    messages.append(AgentMessage(role="assistant", text=draft))
    messages.append(AgentMessage(role="user", text=AGENT_CRITIQUE_RETRIEVAL_DIRECTIVE))

    # Read-only continuation: disable write tools (a write would pause for
    # approval mid-critique) on top of anything the caller already disabled.
    write_tools = {t.name for t in REGISTRY.values() if getattr(t, "requires_approval", False)}
    loop2_disabled = (disabled_tools or set()) | write_tools

    # Budget: continue past loop 1's steps (its tool steps + its answer
    # step), then allow `RAG_CRITIQUE_MAX_STEPS` more (one retrieval + one
    # answer by default). `last_step + 2` skips loop 1's last tool step and
    # its answer step.
    steps_seen = [e["step"] for e in loop1_events if isinstance(e.get("step"), int)]
    last_step = max(steps_seen) if steps_seen else (starting_step - 1)
    next_step = last_step + 2
    crit_steps = max(1, int(settings.SEARCH_ENGINE.get("RAG_CRITIQUE_MAX_STEPS", 2)))

    loop2_events: list[dict[str, Any]] = []
    try:
        pause2 = _drive_loop(
            messages=messages,
            ctx=ctx,
            emit=loop2_events.append,
            run_id=run_id,
            starting_step=next_step,
            seen_sources_by_id=seen_sources_by_id,
            disabled_tools=loop2_disabled,
            system_extra=system_extra,
            trace_hook=trace_hook,
            max_steps=next_step + crit_steps,
            session_id=session_id,
        )
    except Exception:  # noqa: BLE001 — never lose the draft on a critique fault
        log.exception("Critique-with-retrieval continuation failed; keeping draft")
        return list(loop1_events)

    return _merge_critique_events(loop1_events, loop2_events, paused=pause2 is not None)


def _merge_critique_events(
    loop1_events: list[dict[str, Any]],
    loop2_events: list[dict[str, Any]],
    *,
    paused: bool = False,
) -> list[dict[str, Any]]:
    """Decide what to stream after a retrieval-capable critique. Pure (no
    I/O) so it's unit-testable without an LLM.

    Outcomes:
      * paused / loop 2 errored / loop 2 produced no answer → loop 1
        verbatim (the draft is never lost).
      * loop 2 made NO tool call → loop 1 verbatim. No retrieval means no
        new information, so loop 2's paraphrase is discarded to preserve
        the draft's exact wording + citations (the verbatim-KEEP
        guarantee the precision path has).
      * loop 2 retrieved AND answered → loop 1 events minus its
        `answer_delta` + final `done`, then all loop 2 events (whose new
        `sources` + revised answer + `done` supersede on the frontend's
        wholesale-replace).
    """
    loop2_answer = "".join(
        (e.get("text") or "") for e in loop2_events if e.get("type") == "answer_delta"
    )
    loop2_errored = any(e.get("type") == "error" for e in loop2_events)
    loop2_retrieved = any(e.get("type") == "tool_call_start" for e in loop2_events)

    if paused or loop2_errored or not loop2_answer.strip() or not loop2_retrieved:
        return list(loop1_events)

    merged = [e for e in loop1_events if e.get("type") not in ("answer_delta", "done")]
    merged.extend(loop2_events)
    return merged


def _should_abstain_gate(
    tool_results: list[tuple[str, dict[str, Any]]],
    answer: str,
    *,
    paused: bool,
) -> bool:
    """Pure decision for the abstention gate (no I/O — unit-testable).

    Fires only when the turn ATTEMPTED retrieval yet surfaced no
    evidence, and the model didn't already abstain. The evidence rule is
    the load-bearing part:

        had_evidence = (a search_knowledge_base call returned >=1 match)
                       OR (any non-search tool completed without error)

    An empty STRUCTURED result (e.g. `list_tasks` -> no overdue tasks) is
    a grounded "the answer is zero", so a successful non-search tool — even
    with an empty payload — counts as evidence and suppresses the gate.
    Only an empty semantic search with nothing else to lean on is treated
    as "no grounding". Errs toward NOT firing (the safe direction).
    """
    if paused or not answer.strip():
        return False
    if not tool_results:  # zero-tool answer (from context/seed) — leave it
        return False
    had_evidence = any(
        (name == "search_knowledge_base" and bool((result or {}).get("matches")))
        or (name != "search_knowledge_base" and not (result or {}).get("error"))
        for name, result in tool_results
    )
    if had_evidence:
        return False
    return not is_abstention(answer)


def _apply_abstention_to_events(
    events: list[dict[str, Any]], message: str
) -> list[dict[str, Any]]:
    """Rewrite a buffered event stream to replace the answer with `message`.

    Drops the draft `answer_delta`s and any `sources` (there is no genuine
    grounding on the gate path), then injects `message` as a single
    `answer_delta` immediately before `done`. Pure — unit-testable.
    """
    out: list[dict[str, Any]] = []
    injected = False
    for e in events:
        etype = e.get("type")
        if etype in ("answer_delta", "sources"):
            continue
        if etype == "done" and not injected:
            out.append({"type": "answer_delta", "text": message})
            injected = True
        out.append(e)
    if not injected:
        out.append({"type": "answer_delta", "text": message})
        out.append({"type": "done"})
    return out


def _run_with_abstention_gate(
    driver: Callable[
        [
            Callable[[dict[str, Any]], None],
            Callable[[str, dict[str, Any], dict[str, Any]], None] | None,
        ],
        dict[str, Any] | None,
    ],
    emit: Callable[[dict[str, Any]], None],
    outer_trace_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None,
) -> dict[str, Any] | None:
    """Buffer the inner driver, then drop in an honest abstention if the
    turn answered with no grounding (`_should_abstain_gate`).

    Buffering (rather than a live stream filter) is deliberate: a step may
    emit preamble `answer_delta` text *and* still call a tool, so "first
    delta" isn't reliably the final synthesis — only the complete stream
    tells us the answer. Costs TTFT like the self-critique wrapper, hence
    off by default. Composes around either inner driver (plain loop or
    the critique wrapper); a pause descriptor passes straight through.
    """
    captured_events: list[dict[str, Any]] = []
    tool_results: list[tuple[str, dict[str, Any]]] = []

    def _cap_emit(event: dict[str, Any]) -> None:
        captured_events.append(event)

    def _cap_trace(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        tool_results.append((name, result))
        if outer_trace_hook is not None:
            try:
                outer_trace_hook(name, args, result)
            except Exception:  # noqa: BLE001
                log.exception("Outer trace_hook failed inside abstention gate for %s", name)

    pause = driver(_cap_emit, _cap_trace)

    answer = "".join(
        (e.get("text") or "") for e in captured_events if e.get("type") == "answer_delta"
    )
    if _should_abstain_gate(tool_results, answer, paused=pause is not None):
        for e in _apply_abstention_to_events(captured_events, ABSTAIN_MESSAGE):
            emit(e)
    else:
        for e in captured_events:
            emit(e)
    return pause


# Limits for the tool-result blob we hand to the critique LLM. The
# critique only needs the scalar fields (status / due_date / counts) to
# verify the draft — long comment bodies don't carry weight. Mirrors
# the same per-string / per-list caps the eval judge uses, but inlined
# here to keep controller / eval coupling at zero.
_CRITIQUE_MAX_STRING_LEN = 500
_CRITIQUE_MAX_LIST_LEN = 30


def _format_tool_results_for_critique(tool_results: list[dict[str, Any]]) -> str:
    """Compact, size-bounded JSON-ish rendering for the critique prompt."""
    import json  # local — only loaded when the critique fires

    lines: list[str] = []
    for i, tr in enumerate(tool_results, start=1):
        name = tr.get("tool_name") or "?"
        args = json.dumps(tr.get("arguments") or {}, ensure_ascii=False, default=str)
        result = json.dumps(
            _truncate_for_critique(tr.get("result") or {}), ensure_ascii=False, default=str
        )
        lines.append(f"  {i}. {name}({args})\n     result: {result}")
    return "\n".join(lines) if lines else "  (no tool calls)"


def _truncate_for_critique(value: Any) -> Any:
    """Recursively head-tail long strings and cap long lists.
    Scalars (numbers, bools, dates) pass through verbatim — those are
    where the critique's grounding checks land.
    """
    if isinstance(value, str):
        if len(value) > _CRITIQUE_MAX_STRING_LEN:
            half = _CRITIQUE_MAX_STRING_LEN // 2
            return f"{value[:half]} … [{len(value) - _CRITIQUE_MAX_STRING_LEN} chars elided] … {value[-half:]}"
        return value
    if isinstance(value, list):
        truncated = [_truncate_for_critique(v) for v in value[:_CRITIQUE_MAX_LIST_LEN]]
        if len(value) > _CRITIQUE_MAX_LIST_LEN:
            truncated.append(f"… [{len(value) - _CRITIQUE_MAX_LIST_LEN} more items elided]")
        return truncated
    if isinstance(value, dict):
        return {k: _truncate_for_critique(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Resume helpers                                                              #
# --------------------------------------------------------------------------- #


def _rebuild_messages(run: AgentRun) -> tuple[list[AgentMessage], AgentStep | None]:
    """Reconstruct the conversation up to (but not including) the pending step.

    The pending step is the one whose `summary == PENDING_APPROVAL_MARKER`
    and which has neither `result_json` nor `error` filled in yet.
    Returns (messages, pending_step). `pending_step` is None if there's
    no row matching the pending shape — the caller should treat that
    as an error.
    """
    messages: list[AgentMessage] = [_user_turn(run.query)]
    pending: AgentStep | None = None

    for step in run.steps.order_by("step_index", "step_id"):
        # Pending-write rows are skipped here — we hand them back to
        # the caller to resolve.
        if (
            step.tool_name
            and step.summary == PENDING_APPROVAL_MARKER
            and not step.result_json
            and not step.error
        ):
            pending = step
            continue

        # Text-only assistant turns.
        if step.answer_text and not step.tool_name:
            messages.append(AgentMessage(role="assistant", text=step.answer_text))
            continue

        # Completed tool calls (success OR error). Skip rows that have
        # no tool_name — they're error markers like "empty_response".
        if not step.tool_name:
            continue

        fc = FunctionCall(
            name=step.tool_name,
            args=dict(step.arguments_json or {}),
            thought_signature=_coerce_signature(step.thought_signature),
        )
        messages.append(_assistant_function_call_turn(fc))
        if step.error:
            messages.append(_function_response_turn(step.tool_name, {"error": step.error}))
        else:
            messages.append(_function_response_turn(step.tool_name, step.result_json or {}))

    return messages, pending
