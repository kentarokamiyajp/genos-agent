"""Common chunker types.

Each chunker yields `EntityChunks` — one batch per indexable entity
(a chat thread, task, or note). The ingestion pipeline processes one
entity at a time so that stale-chunk deletion can be scoped to that
entity's existing rag_chunks rows.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

# Chat type codes used across the Django app
# (see views/chat/{dm,gm,pm,mdm}_views.py CHAT_TYPE constants).
CHAT_TYPE_DM = 1
CHAT_TYPE_GM = 2
CHAT_TYPE_PM = 3
CHAT_TYPE_MDM = 4

CHAT_TYPE_LABEL = {
    CHAT_TYPE_DM: "dm",
    CHAT_TYPE_GM: "gm",
    CHAT_TYPE_PM: "pm",
    CHAT_TYPE_MDM: "mdm",
}

# Note type codes (mirrors NotePermissionMaster.note_type).
NOTE_TYPE_PERSONAL = 1
NOTE_TYPE_TASK = 2
NOTE_TYPE_CHAT = 3

NOTE_TYPE_LABEL = {
    NOTE_TYPE_PERSONAL: "personal",
    NOTE_TYPE_TASK: "task",
    NOTE_TYPE_CHAT: "chat",
}


@dataclass
class Chunk:
    chunk_id: str
    entity_type: str  # "chat" | "task" | "note"
    entity_id: str  # API-level grouping key, e.g. "chat:dm:42:thread:7"
    chunk_type: str  # "chat_message" | "chat_thread_window" | "task_title_content" | ...
    team_id: str  # UUID stringified
    acl_user_ids: list[str] = field(default_factory=list)

    title: str = ""
    search_text: str = ""
    snippet_text: str = ""
    related_entity_ids: list[str] = field(default_factory=list)

    # Type-specific identifiers (nullable per chunk).
    chat_type: Optional[str] = None  # "dm" | "gm" | "mdm" | "pm"
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    task_id: Optional[str] = None
    note_id: Optional[str] = None
    note_type: Optional[str] = None  # "personal" | "task" | "chat"
    project_id: Optional[str] = None

    # v2 chat-message identity (chat_chunker fills these on focal-
    # message chunks; thread-window chunks leave them None because they
    # aggregate multiple authors).
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    chat_message_id: Optional[str] = None

    # v2 task overlays (task_chunker fills these on every task chunk
    # including comments — comment chunks inherit their parent task's
    # status/assignee so "filter by status" cuts across both).
    task_status: Optional[str] = None
    task_priority: Optional[str] = None
    task_assignee_id: Optional[str] = None
    task_milestone_id: Optional[str] = None
    task_sprint_id: Optional[str] = None

    # v2 note overlays (note_chunker + note_summary_chunker).
    note_owner_id: Optional[str] = None
    note_parent_id: Optional[str] = None

    # spotlight_answer lane only (spotlight_answer_chunker). Stored-only
    # provenance the frontend "Previous answer" card renders — never analyzed
    # for search (search_text already carries Q+A). `answer_text` is the full
    # answer with inline `[type:id]` citation tokens; `answer_sources` is the
    # list of SpotlightResult-shaped source dicts the answer cited.
    answer_text: Optional[str] = None
    answer_sources: Optional[list] = None

    created_at: Optional[str] = None  # ISO 8601
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class EntityChunks:
    """All chunks for a single API-level entity, ready for upsert.

    The ingestion pipeline diffs `chunks` against existing
    `RagChunk` rows scoped to `entity_id`; rows present in the DB but
    missing from `chunks` are deleted from OpenSearch.
    """

    entity_type: str
    entity_id: str
    chunks: list[Chunk]


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def make_snippet(text: str, max_len: int = 280) -> str:
    """Trim search_text down to a UI-friendly snippet."""
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def chat_entity_id(chat_label: str, chat_id, thread_id=None) -> str:
    """Stable entity ID used by the search API to group chat chunks.

    dm:5            (DM-level group, no thread)
    dm:5:thread:9   (thread-level group)
    """
    base = f"{chat_label}:{chat_id}"
    if thread_id is not None:
        return f"{base}:thread:{thread_id}"
    return base
