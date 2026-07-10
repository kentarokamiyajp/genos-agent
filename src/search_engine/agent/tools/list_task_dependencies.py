"""`list_task_dependencies` read tool — a whole scope's blocker graph in one call.

`get_task_blockers` answers "what blocks task X?" one task at a time;
an organize pass over a milestone would need N calls to see the graph.
This tool returns every blocker→blocked edge touching one project or
one milestone in a single call, so the model can order priorities and
due dates around the dependency structure before proposing an
`update_tasks_bulk`.

ACL mirrors `list_tasks`: tenant-scoped, and enumerating a project's
(or a milestone's project's) graph requires project membership.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Q

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneMaster
from origin.models.task.task_models import TaskDependency, TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.list_tasks import _milestone_task_q

_MAX_EDGES = 200


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_project_id = args.get("project_id")
    raw_milestone_id = args.get("milestone_id")
    if (raw_project_id is None) == (raw_milestone_id is None):
        raise ToolError("Pass exactly one of `project_id` or `milestone_id`.")

    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    if raw_project_id is not None:
        try:
            project_id = int(raw_project_id)
        except (TypeError, ValueError):
            raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")
        if project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to list dependencies in project {project_id}. "
                "You are not a member of that project."
            )
        scope_q = Q(blocker_task__project_id=project_id) | Q(blocked_task__project_id=project_id)
        scope_label = f"project {project_id}"
    else:
        try:
            milestone_id = int(raw_milestone_id)
        except (TypeError, ValueError):
            raise ToolError(f"`milestone_id` must be an integer (got {raw_milestone_id!r}).")
        try:
            milestone = MilestoneMaster.objects.get(milestone_id=milestone_id, is_deleted=False)
        except MilestoneMaster.DoesNotExist:
            raise ToolError(f"Milestone {milestone_id} not found.")
        if str(getattr(milestone, "team_id", "") or "") != ctx.team_id:
            raise ToolError("Not authorized: milestone is in a different team.")
        if milestone.project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to list dependencies in milestone {milestone_id}. "
                "You are not a member of that project."
            )
        # Same task-set predicate as the milestone UI rollup / list_tasks,
        # so the graph covers exactly the tasks those surfaces show
        # (direct-FK tasks + subtasks under the backing task).
        member_task_ids = list(
            TaskMaster.objects.filter(_milestone_task_q(milestone)).values_list(
                "task_id", flat=True
            )
        )
        scope_q = Q(blocker_task_id__in=member_task_ids) | Q(
            blocked_task_id__in=member_task_ids
        )
        scope_label = f'milestone "{milestone.title}"'

    edges_qs = (
        TaskDependency.objects.filter(team_id=ctx.team_id)
        .filter(scope_q)
        .select_related("blocker_task__project", "blocked_task__project")
        .order_by("blocked_task_id", "blocker_task_id")[:_MAX_EDGES]
    )

    edges = []
    for dep in edges_qs:
        blocker, blocked = dep.blocker_task, dep.blocked_task
        if blocker.is_deleted or blocked.is_deleted:
            continue
        edges.append(
            {
                "dependency_id": dep.id,
                "blocker_task_id": blocker.task_id,
                "blocker_display_id": blocker.display_id,
                "blocker_title": blocker.title,
                "blocker_status": blocker.status,
                "blocked_task_id": blocked.task_id,
                "blocked_display_id": blocked.display_id,
                "blocked_title": blocked.title,
                "blocked_status": blocked.status,
            }
        )

    return {
        "dependencies": edges,
        "__summary__": f"Found {len(edges)} dependency edge(s) in {scope_label}",
    }


LIST_TASK_DEPENDENCIES = Tool(
    name="list_task_dependencies",
    description=(
        "List every blocker→blocked dependency edge in ONE project or ONE "
        "milestone (pass exactly one of project_id / milestone_id). Use "
        "this — not repeated get_task_blockers calls — when you need the "
        "whole dependency graph, e.g. before proposing update_tasks_bulk "
        "to reprioritize a milestone (blockers of unfinished work "
        "generally deserve earlier due dates and higher priority). "
        "Returns task ids, display ids, titles, and statuses for both "
        "ends of each edge. get_task_blockers remains correct for ONE "
        "task's blockers."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "List all dependency edges touching this project's tasks. "
                    "Resolve via list_projects."
                ),
            },
            "milestone_id": {
                "type": "INTEGER",
                "description": (
                    "List all dependency edges touching this milestone's tasks "
                    "(direct + subtasks). Resolve via list_milestones."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
