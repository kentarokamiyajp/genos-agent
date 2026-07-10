"""Resolve chat-answer citation tokens into app-internal note-body links.

The agent cites workspace entities with a compact token grammar
(`task:12`, `project:5`, `milestone:7`, `note:personal:50`,
`chat:pm:<chat_id>:thread:<thread_id>`). In the CHAT answer the frontend
rewrites those into clickable citations; inside a NOTE body they used to
degrade to plain text, because `blocknote_md` only kept real URLs — so a
"plan note" couldn't link back to its task.

This module closes that gap at save time: `resolve_note_entity_link`
maps a token to the canonical `/workspace/...` route (the same shapes as
the frontend's `sourceToUrl` / routing hooks) plus a fallback label for
bare tokens. Every note editor wraps its DOM in
`useAnchorClickIntercept`, which routes ANY anchor through the
URL-link modal — so a stored relative href is a fully working in-app
link, and still resolves as a normal path if the note is exported.

Scoping: DB-backed types are resolved with a team guard and skip deleted
rows; an unresolvable/foreign token returns None and the caller keeps
the prose (the pre-existing degrade behavior — a note never carries a
dead link). Tokens only ever reference entities the agent read through
ACL-checked tools in the same conversation, and the target views enforce
ACL again on open. Chat tokens are a pure syntactic mapping (they carry
their own UUIDs; the chat view enforces membership). Chat-attached notes
and todo items have no stable deep-link route today → None.
"""

from __future__ import annotations

import re

_TASK_RE = re.compile(r"^task:(\d+)$")
_PROJECT_RE = re.compile(r"^project:(\d+)$")
_MILESTONE_RE = re.compile(r"^milestone:(\d+)$")
_NOTE_RE = re.compile(r"^note:(personal|task):(\d+)$")
_CHAT_RE = re.compile(
    r"^chat:(dm|gm|pm|mdm):([0-9a-fA-F-]{8,36})(?::thread:([0-9a-fA-F-]{8,36}))?$"
)

# Anything shaped like a citation token (mirrors the frontend's
# CITATION_PATTERN vocabulary). Used by blocknote_md to decide whether a
# link target / bare bracket is a candidate for resolution at all.
ENTITY_TOKEN_RE = re.compile(r"^(?:chat|task|note|project|todo|milestone):[^\s()\[\]]+$")


def resolve_note_entity_link(token: str, *, team_id: str) -> tuple[str, str] | None:
    """Token → (href, label), or None to degrade to prose.

    href is app-relative (`/workspace/...`); label is a short fallback
    for bare `[token]` citations (display_id / name / title).
    """
    token = (token or "").strip()

    m = _TASK_RE.match(token)
    if m:
        from origin.models.task.task_models import TaskMaster  # noqa: PLC0415

        task = (
            TaskMaster.objects.filter(task_id=int(m.group(1)), is_deleted=False)
            .select_related("project")
            .first()
        )
        if task is None or str(task.team_id or "") != team_id or not task.project_id:
            return None
        return (
            f"/workspace/tasks/project/{task.project_id}/task/{task.task_id}",
            task.display_id or task.title or f"task {task.task_id}",
        )

    m = _PROJECT_RE.match(token)
    if m:
        from origin.models.project.prj_models import ProjectMaster  # noqa: PLC0415

        project = (
            ProjectMaster.objects.filter(project_id=int(m.group(1)), is_deleted=False)
            .first()
        )
        if project is None or str(project.team_id or "") != team_id:
            return None
        return (
            f"/workspace/tasks/project/{project.project_id}",
            project.project_name or f"project {project.project_id}",
        )

    m = _MILESTONE_RE.match(token)
    if m:
        from origin.models.task.milestone_models import MilestoneMaster  # noqa: PLC0415

        milestone = (
            MilestoneMaster.objects.filter(milestone_id=int(m.group(1)), is_deleted=False)
            .first()
        )
        if milestone is None or str(milestone.team_id or "") != team_id or not milestone.project_id:
            return None
        return (
            f"/workspace/tasks/project/{milestone.project_id}/milestone/{milestone.milestone_id}",
            milestone.title or f"milestone {milestone.milestone_id}",
        )

    m = _NOTE_RE.match(token)
    if m:
        kind, note_id = m.group(1), int(m.group(2))
        if kind == "personal":
            from origin.models.note.personal_note_models import (  # noqa: PLC0415
                PersonalNoteMaster,
            )

            note = (
                PersonalNoteMaster.objects.filter(note_id=note_id)
                .first()
            )
            if note is None or str(note.team_id or "") != team_id:
                return None
            return (f"/workspace/notes/my/{note.note_id}", note.title or f"note {note.note_id}")

        from origin.models.note.task_note_models import TaskNoteMaster  # noqa: PLC0415

        note = (
            TaskNoteMaster.objects.filter(note_id=note_id)
            .first()
        )
        # The task-note route needs all three ids; notes attached only at
        # project level (task_id null) have no deep link → degrade.
        if (
            note is None
            or str(note.team_id or "") != team_id
            or not note.project_id
            or not note.task_id
        ):
            return None
        return (
            f"/workspace/notes/task/project/{note.project_id}/task/{note.task_id}"
            f"/note/{note.note_id}",
            note.title or f"note {note.note_id}",
        )

    m = _CHAT_RE.match(token)
    if m:
        chat_type, chat_id, thread_id = m.group(1), m.group(2), m.group(3)
        href = f"/workspace/chat/{chat_type}/{chat_id}"
        if thread_id:
            href += f"/thread/{thread_id}"
        return (href, "thread" if thread_id else "chat")

    return None
