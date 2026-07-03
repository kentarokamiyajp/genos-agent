"""Task chunker.

Per task we produce up to:

  * `task_title_content` — one chunk holding the title and (for short
    bodies) the full description (`TaskMaster.content`). For long
    bodies the chunk holds title + the first N chars only, and the
    remainder spills into a sibling `task_content_chunk` (see below)
    so a query against a deep-buried keyword can still pull the right
    task.
  * `task_content_chunk` — v2; emitted only when content exceeds the
    `_LONG_CONTENT_THRESHOLD` so short tasks keep one chunk. Carries
    the full content for body-keyword recall while the title chunk
    keeps high-precision title hits.
  * `task_comment` — one chunk per `TaskComments` row (kept separate
    so a question that hits a single comment surfaces *which*
    comment).

v2 — every task chunk (including comments) carries `task_status`,
`task_priority`, `task_assignee_id`, `task_milestone_id`, and
`task_sprint_id` so hybrid search can filter "my open tasks about X"
or rank by status without round-tripping to the DB.

ACL = project members of the task's project. If a task links to a
chat (chat_type/chat_id/thread_id), that chat is added to
`related_entity_ids` so future RAG can pivot from task → discussion.

Skipped: `is_deleted=True` and `is_init_task=True` rows (the latter
are empty placeholders created before a user saves a task).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator, Optional

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskComments, TaskMaster
from origin.search_engine.chunkers.base import (
    CHAT_TYPE_LABEL,
    Chunk,
    EntityChunks,
    chat_entity_id,
    iso,
    make_snippet,
)
from origin.search_engine.text_extraction import extract_text

# v2 — content longer than this triggers the title+content split. Below
# the threshold, one chunk holds title + full content (matches v1
# behavior — no regression on short tasks). Above it, the title chunk
# keeps title + first `_TITLE_CHUNK_CONTENT_HEAD` chars and a sibling
# `task_content_chunk` carries the full content for body recall.
_LONG_CONTENT_THRESHOLD = 1500
_TITLE_CHUNK_CONTENT_HEAD = 400


def iter_task_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    """Yield one EntityChunks per task.

    With `since`, only tasks whose `ts_updated_at` >= since OR whose
    comments' `ts_updated_at` >= since are re-emitted.
    """
    task_qs = TaskMaster.objects.filter(is_deleted=False, is_init_task=False)

    if since is not None:
        dirty_task_ids = set(
            task_qs.filter(ts_updated_at__gte=since).values_list("task_id", flat=True)
        )
        comment_dirty_task_ids = set(
            TaskComments.objects.filter(is_deleted=False, ts_updated_at__gte=since).values_list(
                "task_id", flat=True
            )
        )
        dirty_task_ids |= comment_dirty_task_ids
        task_qs = task_qs.filter(task_id__in=dirty_task_ids)

    task_qs = task_qs.select_related("team", "project")

    task_ids = list(task_qs.values_list("task_id", flat=True))
    project_ids = list(
        task_qs.exclude(project__isnull=True).values_list("project_id", flat=True).distinct()
    )

    # Pre-load ACLs per project.
    members_by_project: dict[int, list[str]] = defaultdict(list)
    for row in ProjectMembers.objects.filter(project_id__in=project_ids).values(
        "project_id", "attendee_id"
    ):
        if row["attendee_id"] is not None:
            members_by_project[row["project_id"]].append(str(row["attendee_id"]))

    # Pre-load comments per task.
    comments_by_task: dict[int, list[TaskComments]] = defaultdict(list)
    for c in TaskComments.objects.filter(task_id__in=task_ids, is_deleted=False).order_by(
        "task_id", "comment_id"
    ):
        comments_by_task[c.task_id].append(c)

    for task in task_qs:
        if not task.team_id:
            continue
        team_id = str(task.team_id)
        project_id = str(task.project_id) if task.project_id else None
        acl_user_ids = members_by_project.get(task.project_id, []) if task.project_id else []
        # Tasks are also legible to the assignee/reporter; include them
        # so they can find tasks they're personally involved in even
        # outside their project membership.
        if task.assignee_id:
            acl_user_ids = list(set(acl_user_ids) | {str(task.assignee_id)})
        if task.reporter_id:
            acl_user_ids = list(set(acl_user_ids) | {str(task.reporter_id)})

        entity_id = f"task:{task.task_id}"
        related = _task_related_ids(task)

        # v2 — task overlay fields shared across the task's chunks.
        # Comment chunks inherit the parent task's status/assignee/etc.
        # so filtering by "Open tasks about X" also turns up the
        # relevant comments.
        task_overlay = {
            "task_status": task.status or None,
            "task_priority": task.priority or None,
            "task_assignee_id": str(task.assignee_id) if task.assignee_id else None,
            "task_milestone_id": str(task.milestone_id) if task.milestone_id else None,
            "task_sprint_id": str(task.sprint_id) if task.sprint_id else None,
        }

        chunks: list[Chunk] = []

        # 1) Title + content chunk(s). Short tasks keep one chunk
        # (v1 behavior); long tasks split into a title-led precision
        # chunk + a content-only recall chunk.
        content_text = extract_text(task.content)
        title_clean = (task.title or "").strip()

        if title_clean or content_text:
            is_long = content_text and len(content_text) > _LONG_CONTENT_THRESHOLD

            # Always emit the title-led chunk. Includes the full content
            # for short tasks; truncated head only for long tasks
            # (the full body lives on the sibling content chunk below).
            if is_long:
                head = content_text[:_TITLE_CHUNK_CONTENT_HEAD].rsplit(" ", 1)[0]
                title_search = "\n".join(p for p in [title_clean, head] if p).strip()
            else:
                title_search = "\n".join(p for p in [title_clean, content_text] if p).strip()

            if title_search:
                chunks.append(
                    Chunk(
                        chunk_id=f"task:{task.task_id}:title_content",
                        entity_type="task",
                        entity_id=entity_id,
                        chunk_type="task_title_content",
                        team_id=team_id,
                        acl_user_ids=acl_user_ids,
                        title=task.title or f"Task {task.task_id}",
                        search_text=title_search,
                        snippet_text=make_snippet(title_search),
                        task_id=str(task.task_id),
                        project_id=project_id,
                        related_entity_ids=related,
                        created_at=iso(task.ts_created_at),
                        updated_at=iso(task.ts_updated_at),
                        **task_overlay,
                    )
                )

            if is_long:
                chunks.append(
                    Chunk(
                        chunk_id=f"task:{task.task_id}:content",
                        entity_type="task",
                        entity_id=entity_id,
                        chunk_type="task_content_chunk",
                        team_id=team_id,
                        acl_user_ids=acl_user_ids,
                        title=task.title or f"Task {task.task_id}",
                        search_text=content_text,
                        snippet_text=make_snippet(content_text),
                        task_id=str(task.task_id),
                        project_id=project_id,
                        related_entity_ids=related,
                        created_at=iso(task.ts_created_at),
                        updated_at=iso(task.ts_updated_at),
                        **task_overlay,
                    )
                )

        # 2) One chunk per comment. Comments inherit the task's
        # status/priority/etc. so a "filter by status=Open" query
        # picks up comments on open tasks too.
        for c in comments_by_task.get(task.task_id, []):
            text = extract_text(c.comment_body)
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"task:{task.task_id}:comment:{c.comment_id}",
                    entity_type="task",
                    entity_id=entity_id,
                    chunk_type="task_comment",
                    team_id=team_id,
                    acl_user_ids=acl_user_ids,
                    title=task.title or f"Task {task.task_id}",
                    search_text=text,
                    snippet_text=make_snippet(text),
                    task_id=str(task.task_id),
                    project_id=project_id,
                    related_entity_ids=related,
                    created_at=iso(c.ts_sent_at),
                    updated_at=iso(c.ts_updated_at),
                    **task_overlay,
                )
            )

        if chunks:
            yield EntityChunks(entity_type="task", entity_id=entity_id, chunks=chunks)


def _task_related_ids(task: TaskMaster) -> list[str]:
    """Return entity-id strings this task points at.

    Uses the same `entity_id` format the chunkers emit so the index
    can match a task's relations to the actual chat/task entities by
    `terms` lookup.
    """
    out: list[str] = []
    chat_type_label = CHAT_TYPE_LABEL.get(task.chat_type) if task.chat_type else None
    if chat_type_label and task.chat_id:
        out.append(chat_entity_id(chat_type_label, task.chat_id, task.thread_id))
    if task.parent_task_id:
        out.append(f"task:{task.parent_task_id}")
    return out
