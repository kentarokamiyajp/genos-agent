"""`list_note_folders` tool — show the user's personal-note organization.

Personal notes ("My Notes") can be organized into sidebar FOLDERS
(`PersonalNoteFolder`, nested arbitrarily deep via `parent_folder_id`);
a root note's `folder_id` files it into one. Folders are personal-only,
owner-scoped, and — importantly — NOT indexed in OpenSearch, so
`search_knowledge_base` returns notes with no folder context. This tool
is the agent's only window into that structure.

It returns each folder WITH the notes filed directly in it (plus the
unfiled top-level notes), so one call yields the actual organization
tree — enough to answer "what's in my Projects folder?" and to resolve a
folder NAME to the `folder_id` that `create_note` / `update_note` need to
file or move a note.

ACL: owner-scoped. Folders and notes are filtered by `ctx.user_id` +
`ctx.team_id` (server-trusted), never by any id in the LLM args. Read
only — no approval.

Only ROOT notes (`parent_note_id IS NULL`) carry a meaningful
`folder_id`; child notes ride along with their root, so they're omitted
here (the tree shows roots).
"""

from __future__ import annotations

from typing import Any

from origin.models.note.personal_note_models import PersonalNoteFolder, PersonalNoteMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext

# Defensive cap on how many notes we enumerate — personal-note counts are
# typically modest, but a pathological account shouldn't produce a huge
# tool payload. Root notes only.
_MAX_NOTES = 500


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    folders = list(
        PersonalNoteFolder.objects.filter(team=ctx.team_id, owner=ctx.user_id)
        .values("folder_id", "parent_folder_id", "name")
        .order_by("name")
    )

    root_notes = list(
        PersonalNoteMaster.objects.filter(
            team=ctx.team_id, owner=ctx.user_id, parent_note_id__isnull=True
        )
        .values("note_id", "title", "folder_id")
        .order_by("title")[:_MAX_NOTES]
    )

    notes_by_folder: dict[int, list[dict[str, Any]]] = {}
    top_level_notes: list[dict[str, Any]] = []
    for n in root_notes:
        entry = {"note_id": n["note_id"], "title": n["title"] or ""}
        if n["folder_id"] is None:
            top_level_notes.append(entry)
        else:
            notes_by_folder.setdefault(n["folder_id"], []).append(entry)

    folder_out = [
        {
            "folder_id": f["folder_id"],
            "name": f["name"],
            "parent_folder_id": f["parent_folder_id"],
            "notes": notes_by_folder.get(f["folder_id"], []),
        }
        for f in folders
    ]

    return {
        "folders": folder_out,
        "top_level_notes": top_level_notes,
        "__summary__": (
            f"{len(folder_out)} personal-note folder(s), "
            f"{len(root_notes)} note(s) ({len(top_level_notes)} unfiled)."
        ),
    }


LIST_NOTE_FOLDERS = Tool(
    name="list_note_folders",
    description=(
        "Show how the user's PERSONAL notes ('My Notes') are organized into "
        "sidebar folders. Returns each folder (id, name, parent_folder_id for "
        "nesting) WITH the notes filed directly in it, plus the unfiled "
        "top-level notes. Use this to answer questions about the user's note "
        "organization ('what's in my Projects folder?') and to resolve a "
        "folder NAME to its numeric folder_id before filing a note with "
        "create_note or moving one with update_note. Folders apply to "
        "personal notes only (not task or chat notes), and are NOT covered by "
        "search_knowledge_base — this tool is the only way to see them."
    ),
    parameters_schema={"type": "OBJECT", "properties": {}},
    run=_run,
    requires_approval=False,
)
