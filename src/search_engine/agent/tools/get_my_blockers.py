"""`get_my_blockers` tool — bidirectional dependency graph scoped to me.

Two halves:
  * `blocked_on_me`  — my open tasks that have an open blocker.
    Answers "what's blocking me?".
  * `blocking_others` — open tasks where I'm the assignee on the blocker
    side, downstream task still open. Answers "who am I holding up?".

Sourced from the same `TaskDependency` model + querysets as
`get_task_blockers`. The per-edge ACL redaction is identical: if the
OTHER endpoint of a dependency lives in a project the caller cannot
see, the edge is dropped and `redacted_count` is incremented so the
agent can mention the graph is partial.

Walks every open task assigned to the caller, then queries dependencies
in two batched lookups (one for `blocker_task_id__in=...`, one for
`blocked_task_id__in=...`) so the cost is O(my-open-tasks + edges) and
not O(my-open-tasks * edges).

ACL contract:
  * Walked tasks: `assignee_id == ctx.user_id` AND project in
    `member_project_ids`.
  * Per-edge: other endpoint's project must be in `member_project_ids`
    or the edge is redacted.
"""

from __future__ import annotations

from typing import Any

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskDependency, TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext

_INACTIVE_STATUSES = ["Closed", "Deleted"]


def _serialize_my_task(t: TaskMaster) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "display_id": t.display_id,
        "title": t.title or "",
        "status": t.status or "",
        "due_date": t.due_date.isoformat() if t.due_date else None,
        "project_id": t.project_id,
        "project_name": t.project.project_name if t.project_id else None,
    }


def _serialize_edge(dep_id: int, other: TaskMaster) -> dict[str, Any]:
    return {
        "dependency_id": dep_id,
        "task_id": other.task_id,
        "display_id": other.display_id,
        "title": other.title or "",
        "status": other.status or "",
        "project_id": other.project_id,
        "project_name": other.project.project_name if other.project_id else None,
        "assignee_id": str(other.assignee_id) if other.assignee_id else None,
    }


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:  # noqa: ARG001
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    my_open_tasks = list(
        TaskMaster.objects.filter(
            team_id=ctx.team_id,
            assignee_id=ctx.user_id,
            project_id__in=member_project_ids,
            is_deleted=False,
            is_init_task=False,
        )
        .exclude(status__in=_INACTIVE_STATUSES)
        .select_related("project")
    )

    if not my_open_tasks:
        return {
            "blocked_on_me_count": 0,
            "blocking_others_count": 0,
            "blocked_on_me": [],
            "blocking_others": [],
            "redacted_count": 0,
            "__summary__": "No open tasks assigned to you.",
        }

    my_task_ids = [t.task_id for t in my_open_tasks]
    task_by_id = {t.task_id: t for t in my_open_tasks}

    # Edges where one of MY tasks is BLOCKED → the OTHER side is the
    # blocker (something I'm waiting on).
    blocked_rows = (
        TaskDependency.objects.filter(blocked_task_id__in=my_task_ids)
        .select_related("blocker_task", "blocker_task__project")
        .exclude(blocker_task__is_deleted=True)
    )
    # Edges where one of MY tasks is BLOCKING → the OTHER side is the
    # blocked task (someone waiting on me).
    blocking_rows = (
        TaskDependency.objects.filter(blocker_task_id__in=my_task_ids)
        .select_related("blocked_task", "blocked_task__project")
        .exclude(blocked_task__is_deleted=True)
    )

    redacted_count = 0
    blocked_on_me_by_task: dict[int, list[dict[str, Any]]] = {}
    blocking_others_by_task: dict[int, list[dict[str, Any]]] = {}

    for dep in blocked_rows:
        blocker = dep.blocker_task
        if blocker is None:
            continue
        # Only surface still-active blockers; a Closed/Deleted blocker is
        # no longer in the user's way.
        if (blocker.status or "") in _INACTIVE_STATUSES:
            continue
        if blocker.project_id and blocker.project_id not in member_project_ids:
            redacted_count += 1
            continue
        blocked_on_me_by_task.setdefault(dep.blocked_task_id, []).append(
            _serialize_edge(dep.id, blocker)
        )

    for dep in blocking_rows:
        blocked = dep.blocked_task
        if blocked is None:
            continue
        if (blocked.status or "") in _INACTIVE_STATUSES:
            continue
        if blocked.project_id and blocked.project_id not in member_project_ids:
            redacted_count += 1
            continue
        blocking_others_by_task.setdefault(dep.blocker_task_id, []).append(
            _serialize_edge(dep.id, blocked)
        )

    blocked_on_me: list[dict[str, Any]] = []
    for task_id, edges in blocked_on_me_by_task.items():
        t = task_by_id.get(task_id)
        if t is None:
            continue
        row = _serialize_my_task(t)
        row["blocked_by"] = edges
        blocked_on_me.append(row)

    blocking_others: list[dict[str, Any]] = []
    for task_id, edges in blocking_others_by_task.items():
        t = task_by_id.get(task_id)
        if t is None:
            continue
        row = _serialize_my_task(t)
        row["blocking"] = edges
        blocking_others.append(row)

    # Stable sort: tasks with the most blockers first (most urgent to
    # unblock); ties broken by task_id for determinism.
    blocked_on_me.sort(key=lambda r: (-len(r["blocked_by"]), r["task_id"]))
    blocking_others.sort(key=lambda r: (-len(r["blocking"]), r["task_id"]))

    summary_bits: list[str] = []
    if blocked_on_me:
        summary_bits.append(f"{len(blocked_on_me)} task(s) blocked")
    if blocking_others:
        summary_bits.append(f"blocking {len(blocking_others)} task(s) for others")
    if not summary_bits:
        summary_bits.append("no active dependencies")
    if redacted_count:
        summary_bits.append(f"{redacted_count} edge(s) redacted")

    return {
        "blocked_on_me_count": len(blocked_on_me),
        "blocking_others_count": len(blocking_others),
        "blocked_on_me": blocked_on_me,
        "blocking_others": blocking_others,
        "redacted_count": redacted_count,
        "__summary__": "; ".join(summary_bits),
    }


GET_MY_BLOCKERS = Tool(
    name="get_my_blockers",
    description=(
        "Bidirectional dependency graph for the current user's open "
        "tasks. Returns two lists: `blocked_on_me` (my open tasks with "
        "still-active blockers) and `blocking_others` (open tasks I am "
        "the blocker for and that are still open). Use for 'what's "
        "blocking me?', 'am I blocked on anything?', 'who am I holding "
        "up?', or 'what should others know I owe them?'. Each edge "
        "includes the other task's display_id, title, status, project, "
        "and assignee. Cross-project edges to projects the caller "
        "cannot see are dropped and counted in `redacted_count` — "
        "mention the partial-graph caveat to the user when "
        "`redacted_count > 0`."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
    run=_run,
)
