"""`get_task_blockers` tool — dependency graph for one task.

Returns both directions of the dependency edges:
  * `blocked_by` — rows where the target task is the BLOCKED side,
    so the "other" task is the one blocking progress.
  * `blocking`   — rows where the target task is the BLOCKER, so
    the "other" task is downstream of it.

Milestones use their backing `TaskMaster` (`is_milestone=True`) in the
dependency table, so passing a milestone's backing-task `task_id` works
transparently — the model can ask "is milestone X blocked?" using the
same tool.

ACL contract:
  * Tenant guard: target task's team must equal `ctx.team_id`.
  * Visibility: caller must be in `task_acl_user_ids(target)` (project
    member, assignee, or reporter) — same gate as `fetch_task`.
  * Per-edge filter: dependency rows where the OTHER endpoint lives in
    a project the caller is NOT a member of are DROPPED, not returned.
    `redacted_count` reports how many were hidden so the LLM can tell
    the user the graph is partial without enumerating inaccessible
    task identities.
"""

from __future__ import annotations

from typing import Any

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskDependency, TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError


def _hydrate_ref(dep_id: int, other: TaskMaster) -> dict[str, Any]:
    """Shape one endpoint for the LLM. Drops UI-only fields (colour chips)."""
    return {
        "dependency_id": dep_id,
        "task_id": other.task_id,
        "display_id": other.display_id,
        "title": other.title or "",
        "status": other.status or "",
        "project_id": other.project_id,
        "project_name": other.project.project_name if other.project_id else None,
        "is_milestone": bool(other.is_milestone),
        "assignee_id": str(other.assignee_id) if other.assignee_id else None,
    }


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_task_id = args.get("task_id")
    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        raise ToolError(f"`task_id` must be an integer (got {raw_task_id!r}).")

    try:
        target = TaskMaster.objects.select_related("project").get(
            task_id=task_id, is_deleted=False
        )
    except TaskMaster.DoesNotExist:
        raise ToolError(f"Task {task_id} not found.")

    if str(getattr(target, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: task is in a different team.")
    allowed = task_acl_user_ids(
        getattr(target, "project_id", None),
        getattr(target, "assignee_id", None),
        getattr(target, "reporter_id", None),
    )
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to view task {task_id}.")

    # Per-edge ACL set: projects the caller can see at all. The OTHER
    # endpoint of each dependency must live in one of these projects;
    # cross-project edges to a project the caller isn't in are redacted.
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    # blocking[] = rows where target IS the blocker; "other" is blocked.
    # blocked_by[] = rows where target IS the blocked; "other" is blocker.
    # Mirrors TaskDependencyView.get logic; soft-deleted endpoints are dropped
    # so the agent never reports ghosts.
    blocking_rows = (
        TaskDependency.objects.filter(blocker_task_id=task_id)
        .select_related("blocked_task", "blocked_task__project")
        .exclude(blocked_task__is_deleted=True)
    )
    blocked_by_rows = (
        TaskDependency.objects.filter(blocked_task_id=task_id)
        .select_related("blocker_task", "blocker_task__project")
        .exclude(blocker_task__is_deleted=True)
    )

    blocking: list[dict[str, Any]] = []
    blocked_by: list[dict[str, Any]] = []
    redacted_count = 0

    for dep in blocking_rows:
        other = dep.blocked_task
        if other is None:
            continue
        if other.project_id and other.project_id not in member_project_ids:
            redacted_count += 1
            continue
        blocking.append(_hydrate_ref(dep.id, other))

    for dep in blocked_by_rows:
        other = dep.blocker_task
        if other is None:
            continue
        if other.project_id and other.project_id not in member_project_ids:
            redacted_count += 1
            continue
        blocked_by.append(_hydrate_ref(dep.id, other))

    summary_bits: list[str] = []
    if blocked_by:
        summary_bits.append(f"blocked by {len(blocked_by)}")
    if blocking:
        summary_bits.append(f"blocking {len(blocking)}")
    if not summary_bits:
        summary_bits.append("no dependencies")
    if redacted_count:
        summary_bits.append(f"{redacted_count} redacted")

    return {
        "task_id": target.task_id,
        "display_id": target.display_id,
        "title": target.title or "",
        "is_milestone": bool(target.is_milestone),
        "blocked_by": blocked_by,
        "blocking": blocking,
        "redacted_count": redacted_count,
        "__summary__": f"Task {target.display_id}: " + ", ".join(summary_bits),
    }


GET_TASK_BLOCKERS = Tool(
    name="get_task_blockers",
    description=(
        "Return the dependency graph edges for one task: which tasks block "
        "it (`blocked_by`) and which tasks it blocks (`blocking`). Works "
        "for milestones too (they share the same `TaskDependency` table "
        "via their backing task). Use for 'is this task blocked?', 'what "
        "does this task block?', or 'who's waiting on this milestone?'. "
        "Each edge includes the other task's display_id, title, status, "
        "and project. Dependencies pointing at projects the caller can't "
        "see are dropped and counted in `redacted_count` — the agent "
        "should mention the partial-graph caveat to the user when "
        "redacted_count > 0."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "task_id": {
                "type": "INTEGER",
                "description": (
                    "Numeric task id (or milestone's backing task id). "
                    "Resolve a display_id like 'WRD-5' with "
                    "search_knowledge_base or list_tasks first."
                ),
            },
        },
        "required": ["task_id"],
    },
    run=_run,
)
