"""`add_comment` write tool — Phase 11.

Add a comment to an existing task. `TaskComments.comment_id` is a
per-task sequence (not a global auto-increment), so we compute the
next id ourselves via `count + 1` — same approach as the existing
`TaskCommentsView` in `views/task/task_views.py`.

ACL: same as `fetch_task` (project members + assignee + reporter).
A user who can read the task's discussion can also contribute to it.

Approval flow (Phase 7): `requires_approval=True`. The user sees the
proposed comment body in the Approve / Reject card before it lands.
"""

from __future__ import annotations

import uuid
from typing import Any

from django.db import transaction

from origin.models.task.task_models import TaskComments, TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_PARA_PROPS = {
    "textColor": "default",
    "textAlignment": "left",
    "backgroundColor": "default",
}


def _paragraph(text: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "paragraph",
        "props": dict(_PARA_PROPS),
        "content": ([{"text": text, "type": "text", "styles": {}}] if text else []),
        "children": [],
    }


def _wrap_blocknote(text: str) -> list[dict[str, Any]]:
    """Emit the exact BlockNote shape the chat preview expects.

    Real user-typed comments persist as: one paragraph block per text
    line, followed by a trailing blank paragraph (BlockNote's editor
    sentinel). `BnChatPreview` renders comments via
    `initialContent: content.slice(0, -1)`, so a single-block body
    becomes an empty array and BlockNote throws "initialContent must
    be a non-empty array of blocks". We must emit the trailing blank
    block.
    """
    if not text:
        return []
    lines = text.split("\n")
    return [_paragraph(line) for line in lines] + [_paragraph("")]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_task_id = args.get("task_id")
    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        raise ToolError(f"`task_id` must be an integer (got {raw_task_id!r}).")

    body_text = (args.get("body_text") or "").strip()
    if not body_text:
        raise ToolError("`body_text` is required and must be non-empty.")

    try:
        task = TaskMaster.objects.get(task_id=task_id)
    except TaskMaster.DoesNotExist:
        raise ToolError(f"Task {task_id} not found.")
    if task.is_deleted:
        raise ToolError(f"Task {task_id} has been deleted.")

    if str(getattr(task, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: task is in a different team.")
    allowed = task_acl_user_ids(
        getattr(task, "project_id", None),
        getattr(task, "assignee_id", None),
        getattr(task, "reporter_id", None),
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to comment on task {task_id}.")

    # `comment_id` is per-task (UniqueConstraint(task, comment_id)).
    # Wrap the count + insert in a transaction so concurrent inserts
    # can't collide on the same id. SELECT-then-INSERT is racy without
    # this; with the transaction the unique constraint will surface
    # any remaining race as a ToolError on .save().
    try:
        with transaction.atomic():
            next_id = TaskComments.objects.select_for_update().filter(task=task).count() + 1
            comment = TaskComments.objects.create(
                task=task,
                sender_id=ctx.user_id,
                comment_id=next_id,
                comment_body=_wrap_blocknote(body_text),
            )
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"Failed to add comment: {e}")

    return {
        "task_id": task_id,
        "comment_id": comment.comment_id,
        "__summary__": (f"Added comment #{comment.comment_id} to task {task.display_id}"),
    }


ADD_COMMENT = Tool(
    name="add_comment",
    description=(
        "Add a plain-text comment to an existing task. REQUIRES USER "
        "APPROVAL — the user sees your proposed comment text before it's "
        "posted. Required: task_id, body_text. Use this when the user "
        "asks you to leave a note, post a comment, or reply on a task. "
        "Use `fetch_task` first if you need to read existing comments "
        "before composing yours."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "task_id": {
                "type": "INTEGER",
                "description": "Numeric task id to comment on.",
            },
            "body_text": {
                "type": "STRING",
                "description": (
                    "Plain-text comment body. Keep it concise; mentions "
                    "(@user) and rich formatting are not supported by this "
                    "tool — users add those through the UI."
                ),
            },
        },
        "required": ["task_id", "body_text"],
    },
    run=_run,
    requires_approval=True,
)
