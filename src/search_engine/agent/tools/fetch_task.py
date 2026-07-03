"""`fetch_task` tool — load one task with its comments.

Direct ORM access (no HTTP roundtrip to GetTaskView). Returns the
minimal LLM-friendly fields: title, status, content text, last N
comments. The existing detail endpoint returns base64 attachments
and UI-formatted dates — none of which help grounding.

ACL: requesting user must be in the task's project OR be the
assignee / reporter — same set the chunker stamps onto each chunk's
`acl_user_ids` field.
"""

from __future__ import annotations

from typing import Any

from origin.models.task.task_models import TaskComments, TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    wrap_workspace_content,
)
from origin.search_engine.text_extraction import extract_text

_COMMENTS_CAP = 20


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_task_id = args.get("task_id")
    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        raise ToolError(f"task_id must be an integer (got {raw_task_id!r}).")

    try:
        # select_related("project") lets the display_id property
        # ("<project.code>-<project_task_number>") resolve without an
        # extra query.
        task = TaskMaster.objects.select_related("project").get(task_id=task_id)
    except TaskMaster.DoesNotExist:
        raise ToolError(f"Task {task_id} not found.")

    if task.is_deleted:
        raise ToolError(f"Task {task_id} has been deleted.")

    # Tenant guard first — never leak data across teams even if the
    # within-team ACL check would let the user through.
    if str(getattr(task, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: task is in a different team.")

    allowed = task_acl_user_ids(
        getattr(task, "project_id", None),
        getattr(task, "assignee_id", None),
        getattr(task, "reporter_id", None),
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to read task {task_id}.")

    content_text = extract_text(task.content)

    # Most-recent N comments. The chunker indexes all comments, but the
    # LLM rarely needs full history — last 20 is plenty for grounding.
    comments_qs = TaskComments.objects.filter(task=task, is_deleted=False).order_by("-comment_id")[
        :_COMMENTS_CAP
    ]
    comments = []
    for c in reversed(list(comments_qs)):  # back into chronological order
        text = extract_text(c.comment_body)
        if not text:
            continue
        comments.append(
            {
                "comment_id": c.comment_id,
                "sender_id": str(getattr(c, "sender_id", "") or ""),
                "text": wrap_workspace_content(text),
                "ts": c.ts_sent_at.isoformat() if c.ts_sent_at else None,
            }
        )

    return {
        "task_id": task_id,
        "display_id": task.display_id,
        "title": task.title or "",
        "status": task.status,
        "priority": task.priority,
        "effort_level": task.effort_level,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "project_id": str(task.project_id) if task.project_id else None,
        "assignee_id": str(task.assignee_id) if task.assignee_id else None,
        "reporter_id": str(task.reporter_id) if task.reporter_id else None,
        "content_text": wrap_workspace_content(content_text),
        "comments": comments,
        "__summary__": (
            f"Loaded task {task.display_id}" + (f" + {len(comments)} comments" if comments else "")
        ),
    }


FETCH_TASK = Tool(
    name="fetch_task",
    description=(
        "Load the full content of one task: title, description text, "
        "status, priority, assignee/reporter, and the most recent "
        "comments. Use after `search_knowledge_base` if the snippet "
        "isn't enough and you need to read the task body or its "
        "discussion. ACL is enforced — only tasks the user can access "
        "are returned."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "task_id": {
                "type": "INTEGER",
                "description": "Numeric task id (e.g. 123).",
            },
        },
        "required": ["task_id"],
    },
    run=_run,
)
