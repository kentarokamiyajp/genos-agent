"""`create_task_plan` composite write tool — a whole plan in ONE approval.

Creates a milestone plus its task tree (nesting + blocker dependencies)
— or a batch of tasks under an existing milestone, sub-tasks of an
existing task (`parent_task_id`, the "break this task down from its
comments" flow), or standalone — in a single approval-gated call. This is the tool behind "create a milestone
and tasks based on this chat": without it the model would need one
`create_task` proposal per task, which means one Approve click per task
and no way to express nesting or dependencies at all (`create_task`
cannot set `milestone` / `parent_task_id`).

Semantics mirror the UI paths exactly:
  * Milestone creation goes through `milestone_service.create_milestone`
    (same backing-task invariants as `MilestoneView.post`).
  * Top-level tasks in milestone mode get `parent_task_id` = the
    milestone's backing task AND the `milestone` FK — the same
    double-link `TaskMasterView.post`'s milestone↔parent bridge and the
    demo seeder produce.
  * Sub-tasks (`parent_index`) nest under an earlier task in the batch
    and inherit its milestone FK.
  * Tasks are created in array order, parents first, so the
    `root_task_id` post_save signal resolves each child in O(1).

Everything runs inside one transaction: any failure rolls the whole
plan back — the user approved a specific plan, so a partially-created
one is worse than a clean error the model can retry.

Intra-batch references are by 0-based array index (`parent_index`,
`blocked_by_indexes`) because the real task_ids don't exist until
execution. Validation is all-up-front and names the offending index so
the model can self-correct after a rejection or error.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from django.db import transaction

from origin.models.common.team_models import TeamMembers
from origin.models.project.prj_models import ProjectMaster
from origin.models.task.milestone_models import MilestoneMaster
from origin.models.task.task_models import TaskDependency, TaskMaster
from origin.search_engine.agent.acl import task_acl_user_ids
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.blocknote_md import markdown_to_blocks
from origin.search_engine.agent.tools.task_enums import (
    EFFORT_ENUM,
    PRIORITY_ENUM,
    VALID_EFFORTS,
    VALID_PRIORITIES,
)
from origin.services.milestone_service import create_milestone, ensure_backing_task
from origin.services.task_cache import invalidate_project_tasks_cache

# Hard cap on tasks per plan. Keeps the single function-call emission
# well under model output limits and the approval card reviewable; the
# system prompt tells the model to split larger plans.
_MAX_PLAN_TASKS = 20


def _parse_date(raw: Any, where: str, errors: list[str]) -> date | None:
    if raw in (None, ""):
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        errors.append(f"{where}: invalid ISO date {raw!r} (expected YYYY-MM-DD).")
        return None


def _check_enum(raw: Any, valid: set[str], where: str, field: str, errors: list[str]):
    if raw is None or raw == "":
        return None
    if raw not in valid:
        errors.append(f"{where}: `{field}` must be one of {sorted(valid)} (got {raw!r}).")
        return None
    return raw


def _find_dependency_cycle(edges_by_node: dict[int, list[int]]) -> list[int] | None:
    """DFS over blocked→blocker edges; returns one cycle path or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(edges_by_node, WHITE)
    stack_path: list[int] = []

    def visit(node: int) -> list[int] | None:
        color[node] = GRAY
        stack_path.append(node)
        for nxt in edges_by_node.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                return stack_path[stack_path.index(nxt) :] + [nxt]
            if color.get(nxt, WHITE) == WHITE:
                found = visit(nxt)
                if found:
                    return found
        stack_path.pop()
        color[node] = BLACK
        return None

    for start in edges_by_node:
        if color[start] == WHITE:
            found = visit(start)
            if found:
                return found
    return None


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # ---- project + ACL (same layering as create_task) ----
    raw_project_id = args.get("project_id")
    try:
        project_id = int(raw_project_id)
    except (TypeError, ValueError):
        raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")

    try:
        project = ProjectMaster.objects.get(project_id=project_id, is_deleted=False)
    except ProjectMaster.DoesNotExist:
        raise ToolError(f"Project {project_id} not found.")

    if str(getattr(project, "team_id", "") or "") != ctx.team_id:
        raise ToolError("Not authorized: project belongs to a different team.")

    allowed = task_acl_user_ids(project_id=project_id, assignee_id=None, reporter_id=None)
    if ctx.user_id not in allowed:
        raise ToolError(f"Not authorized to create tasks in project {project_id}.")

    # ---- attach-mode resolution (new milestone XOR existing milestone
    # XOR sub-tasks of an existing task XOR standalone) ----
    milestone_spec = args.get("milestone")
    raw_existing_id = args.get("existing_milestone_id")
    raw_parent_task_id = args.get("parent_task_id")
    modes_given = sum(
        x is not None for x in (milestone_spec, raw_existing_id, raw_parent_task_id)
    )
    if modes_given > 1:
        raise ToolError(
            "Pass at most ONE of `milestone` (create new), `existing_milestone_id`, "
            "or `parent_task_id` (sub-tasks of an existing task)."
        )

    parent_task: TaskMaster | None = None
    if raw_parent_task_id is not None:
        try:
            parent_id = int(raw_parent_task_id)
        except (TypeError, ValueError):
            raise ToolError(f"`parent_task_id` must be an integer (got {raw_parent_task_id!r}).")
        try:
            parent_task = TaskMaster.objects.select_related("project").get(task_id=parent_id)
        except TaskMaster.DoesNotExist:
            raise ToolError(f"Task {parent_id} not found.")
        if parent_task.is_deleted:
            raise ToolError(f"Task {parent_id} has been deleted.")
        if str(getattr(parent_task, "team_id", "") or "") != ctx.team_id:
            raise ToolError("Not authorized: parent task is in a different team.")
        if parent_task.project_id != project_id:
            raise ToolError(
                f"Task {parent_id} belongs to project {parent_task.project_id}, "
                f"not {project_id}."
            )

    existing_milestone: MilestoneMaster | None = None
    if raw_existing_id is not None:
        try:
            existing_id = int(raw_existing_id)
        except (TypeError, ValueError):
            raise ToolError(f"`existing_milestone_id` must be an integer (got {raw_existing_id!r}).")
        try:
            existing_milestone = MilestoneMaster.objects.select_related("task").get(
                milestone_id=existing_id, is_deleted=False
            )
        except MilestoneMaster.DoesNotExist:
            raise ToolError(f"Milestone {existing_id} not found.")
        if str(getattr(existing_milestone, "team_id", "") or "") != ctx.team_id:
            raise ToolError("Not authorized: milestone belongs to a different team.")
        if existing_milestone.project_id != project_id:
            raise ToolError(
                f"Milestone {existing_id} belongs to project "
                f"{existing_milestone.project_id}, not {project_id}."
            )

    # ---- validate the whole plan up-front, collecting every problem ----
    errors: list[str] = []

    if milestone_spec is not None:
        if not isinstance(milestone_spec, dict):
            raise ToolError("`milestone` must be an object.")
        if not (milestone_spec.get("title") or "").strip():
            errors.append("milestone: `title` is required.")
        _check_enum(milestone_spec.get("priority"), VALID_PRIORITIES, "milestone", "priority", errors)
        _check_enum(
            milestone_spec.get("effort_level"), VALID_EFFORTS, "milestone", "effort_level", errors
        )
        _parse_date(milestone_spec.get("start_date"), "milestone.start_date", errors)
        _parse_date(milestone_spec.get("due_date"), "milestone.due_date", errors)

    tasks = args.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ToolError("`tasks` must be a non-empty array.")
    if len(tasks) > _MAX_PLAN_TASKS:
        raise ToolError(
            f"Too many tasks: {len(tasks)} (max {_MAX_PLAN_TASKS} per plan). "
            "Split the plan and tell the user."
        )

    # Collect every referenced user id for one batched membership check.
    assignee_ids: set[str] = set()
    if milestone_spec:
        for uid in milestone_spec.get("assignee_ids") or []:
            if uid:
                assignee_ids.add(str(uid))

    edges_by_node: dict[int, list[int]] = {i: [] for i in range(len(tasks))}
    for i, spec in enumerate(tasks):
        where = f"tasks[{i}]"
        if not isinstance(spec, dict):
            errors.append(f"{where}: must be an object.")
            continue
        if not (spec.get("title") or "").strip():
            errors.append(f"{where}: `title` is required.")
        _check_enum(spec.get("priority"), VALID_PRIORITIES, where, "priority", errors)
        _check_enum(spec.get("effort_level"), VALID_EFFORTS, where, "effort_level", errors)
        _parse_date(spec.get("start_date"), f"{where}.start_date", errors)
        _parse_date(spec.get("due_date"), f"{where}.due_date", errors)
        if spec.get("assignee_id"):
            assignee_ids.add(str(spec["assignee_id"]))

        parent_index = spec.get("parent_index")
        if parent_index is not None:
            try:
                parent_index = int(parent_index)
            except (TypeError, ValueError):
                errors.append(f"{where}: `parent_index` must be an integer.")
                parent_index = None
        if parent_index is not None:
            if not (0 <= parent_index < i):
                errors.append(
                    f"{where}: `parent_index` must reference an EARLIER task "
                    f"(0..{i - 1}), got {parent_index}. Order parents before children."
                )
            elif isinstance(tasks[parent_index], dict) and tasks[parent_index].get(
                "parent_index"
            ) is not None:
                errors.append(
                    f"{where}: parent tasks[{parent_index}] is itself a sub-task — "
                    "only one level of nesting below top-level tasks is supported."
                )

        seen_blockers: set[int] = set()
        for raw_b in spec.get("blocked_by_indexes") or []:
            try:
                b = int(raw_b)
            except (TypeError, ValueError):
                errors.append(f"{where}: `blocked_by_indexes` entries must be integers.")
                continue
            if not (0 <= b < len(tasks)):
                errors.append(
                    f"{where}: `blocked_by_indexes` entry {b} is out of range "
                    f"(0..{len(tasks) - 1})."
                )
                continue
            if b == i:
                errors.append(f"{where}: a task cannot be blocked by itself.")
                continue
            if b in seen_blockers:
                continue
            seen_blockers.add(b)
            edges_by_node[i].append(b)

    cycle = _find_dependency_cycle(edges_by_node)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(str(n) for n in cycle))

    if assignee_ids:
        # Every referenced user must be an active member of THIS team —
        # scoped to ctx.team_id so the model cannot smuggle in a foreign
        # tenant's UUID (same layering as assign_task).
        member_ids = {
            str(a)
            for a in TeamMembers.objects.filter(
                team_id=ctx.team_id,
                attendee_id__in=list(assignee_ids),
                is_deleted=False,
            ).values_list("attendee_id", flat=True)
        }
        for missing in sorted(assignee_ids - member_ids):
            errors.append(
                f"assignee {missing!r} is not an active member of this team. "
                "Use get_team_members to find valid user ids."
            )

    if errors:
        raise ToolError("Plan validation failed:\n- " + "\n- ".join(errors[:25]))

    # ---- execute: all-or-nothing ----
    try:
        with transaction.atomic():
            milestone: MilestoneMaster | None = None
            if milestone_spec is not None:
                milestone = create_milestone(
                    project,
                    reporter_id=ctx.user_id,
                    title=milestone_spec["title"].strip(),
                    description_blocks=markdown_to_blocks(
                        (milestone_spec.get("description_markdown") or "").strip()
                    ),
                    priority=milestone_spec.get("priority") or None,
                    effort_level=milestone_spec.get("effort_level") or None,
                    start_date=milestone_spec.get("start_date") or None,
                    due_date=milestone_spec.get("due_date") or None,
                    assignee_ids=milestone_spec.get("assignee_ids") or [],
                )
            elif existing_milestone is not None:
                milestone = existing_milestone
                ensure_backing_task(milestone)

            if parent_task is not None:
                # Sub-tasks of an existing task: nest under it and
                # inherit its milestone/sprint (same semantics a
                # UI-created sub-task gets).
                anchor_parent_id = parent_task.task_id
                anchor_milestone = parent_task.milestone
                sprint_id = parent_task.sprint_id
            else:
                anchor_parent_id = milestone.task_id if milestone is not None else None
                anchor_milestone = milestone
                sprint_id = milestone.sprint_id if milestone is not None else None

            created: list[TaskMaster] = []
            for spec in tasks:
                parent_index = spec.get("parent_index")
                if parent_index is not None:
                    parent = created[int(parent_index)]
                    parent_task_id = parent.task_id
                    task_milestone = parent.milestone
                else:
                    parent_task_id = anchor_parent_id
                    task_milestone = anchor_milestone
                created.append(
                    TaskMaster.objects.create(
                        team_id=ctx.team_id,
                        project=project,
                        milestone=task_milestone,
                        sprint_id=sprint_id,
                        reporter_id=ctx.user_id,
                        assignee_id=str(spec["assignee_id"]) if spec.get("assignee_id") else None,
                        title=spec["title"].strip(),
                        content=markdown_to_blocks((spec.get("content_markdown") or "").strip()),
                        status="Open",
                        priority=spec.get("priority") or None,
                        effort_level=spec.get("effort_level") or None,
                        start_date=spec.get("start_date") or None,
                        due_date=spec.get("due_date") or None,
                        parent_task_id=parent_task_id,
                    )
                )

            dependencies = [
                TaskDependency(
                    blocker_task_id=created[b].task_id,
                    blocked_task_id=created[i].task_id,
                    team_id=ctx.team_id,
                    created_by_id=ctx.user_id,
                )
                for i, blockers in edges_by_node.items()
                for b in blockers
            ]
            if dependencies:
                TaskDependency.objects.bulk_create(dependencies, ignore_conflicts=True)
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001 — surface as ToolError for the model
        raise ToolError(f"Failed to create the plan (nothing was created): {e}")

    # The backing task (and every plan task) is a new row in the project
    # task table — drop the cached listing after commit.
    invalidate_project_tasks_cache(ctx.team_id, project_id)

    milestone_out = None
    if milestone is not None:
        milestone_out = {
            "milestone_id": milestone.milestone_id,
            "task_id": milestone.task_id,
            "display_id": milestone.task.display_id if milestone.task_id else None,
            "title": milestone.title,
        }

    tasks_out = [
        {
            "index": i,
            "task_id": t.task_id,
            "display_id": t.display_id,
            "title": t.title,
            "parent_task_id": t.parent_task_id,
        }
        for i, t in enumerate(created)
    ]

    parent_out = None
    if parent_task is not None:
        parent_out = {
            "task_id": parent_task.task_id,
            "display_id": parent_task.display_id,
            "title": parent_task.title,
        }

    n_deps = len(dependencies)
    if milestone_spec is not None:
        summary = (
            f'Created milestone "{milestone.title}" with {len(created)} task(s)'
            + (f", {n_deps} dependency(ies)" if n_deps else "")
        )
    elif parent_task is not None:
        summary = f"Created {len(created)} sub-task(s) under {parent_task.display_id}" + (
            f", {n_deps} dependency(ies)" if n_deps else ""
        )
    else:
        target = f' in milestone "{milestone.title}"' if milestone is not None else ""
        summary = f"Created {len(created)} task(s){target}" + (
            f", {n_deps} dependency(ies)" if n_deps else ""
        )

    return {
        "milestone": milestone_out,
        "parent_task": parent_out,
        "project_id": project_id,
        "project_name": project.project_name,
        "tasks": tasks_out,
        "dependencies_created": n_deps,
        "__summary__": summary,
    }


_TASK_ITEM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "description": "Short task title (1 line)."},
        "content_markdown": {
            "type": "STRING",
            "description": (
                "Optional task body in markdown following the house task "
                "template: '### 🧾 Summary', '### 🪜 Motivation', "
                "'### ✅ Acceptance criteria' (bulleted). For bug-type tasks "
                "use '### 🐞 Summary', '### 🔁 Steps to reproduce', "
                "'### 🎯 Expected behavior', '### 💥 Actual behavior'. "
                "Keep it under ~150 words."
            ),
        },
        "priority": {
            "type": "STRING",
            "enum": PRIORITY_ENUM,
            "description": "Optional priority.",
        },
        "effort_level": {
            "type": "STRING",
            "enum": EFFORT_ENUM,
            "description": "Optional effort estimate.",
        },
        "start_date": {"type": "STRING", "description": "Optional ISO date (YYYY-MM-DD)."},
        "due_date": {"type": "STRING", "description": "Optional ISO date (YYYY-MM-DD)."},
        "assignee_id": {
            "type": "STRING",
            "description": (
                "Optional assignee user UUID from get_team_members / "
                "list_project_members. Omit for unassigned."
            ),
        },
        "parent_index": {
            "type": "INTEGER",
            "description": (
                "Optional 0-based index of an EARLIER task in this array to "
                "nest under as a sub-task. The parent must be a top-level "
                "task (one nesting level). Order parents before children."
            ),
        },
        "blocked_by_indexes": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "description": (
                "Optional 0-based indexes of tasks in this array that must "
                "finish before this one can start (blocker -> blocked edges)."
            ),
        },
    },
    "required": ["title"],
}

CREATE_TASK_PLAN = Tool(
    name="create_task_plan",
    description=(
        "Create a whole work plan in ONE approval: a new milestone plus its "
        "tasks/sub-tasks with dependencies — or a batch of tasks attached to "
        "an existing milestone (existing_milestone_id), nested as SUB-TASKS "
        "of an existing task (parent_task_id — e.g. 'break this task into "
        "sub-tasks based on its comments'), or standalone in a project. "
        "REQUIRES USER APPROVAL — the user sees the full proposed plan and "
        "approves once. Use this whenever the user asks to create a "
        "milestone with tasks, break a discussion/chat/task into tasks, or "
        "add MULTIPLE related tasks — NEVER a series of create_task calls. "
        "(create_task remains correct for one single ad-hoc task; for a "
        "plan as a DOCUMENT/note rather than tasks, use create_note.) Max "
        f"{_MAX_PLAN_TASKS} tasks; reference other tasks in the batch by "
        "array index (parent_index, blocked_by_indexes)."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Numeric project id the plan belongs to. Resolve via "
                    "list_projects first; if ambiguous, ASK the user which "
                    "project before proposing."
                ),
            },
            "milestone": {
                "type": "OBJECT",
                "description": (
                    "Optional NEW milestone to create; every top-level task "
                    "in `tasks` is attached to it. Mutually exclusive with "
                    "existing_milestone_id. Omit both for standalone tasks."
                ),
                "properties": {
                    "title": {"type": "STRING", "description": "Milestone title."},
                    "description_markdown": {
                        "type": "STRING",
                        "description": (
                            "Optional milestone body in markdown following "
                            "the house milestone template: '### 🎯 Goal', "
                            "'### ✅ Success criteria' (bulleted), "
                            "'### 📦 In scope', '### 🚫 Out of scope', "
                            "'### ⚠️ Risks & dependencies'."
                        ),
                    },
                    "priority": {"type": "STRING", "enum": PRIORITY_ENUM},
                    "effort_level": {"type": "STRING", "enum": EFFORT_ENUM},
                    "start_date": {
                        "type": "STRING",
                        "description": "Optional ISO date (YYYY-MM-DD).",
                    },
                    "due_date": {
                        "type": "STRING",
                        "description": "Optional ISO date (YYYY-MM-DD).",
                    },
                    "assignee_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "Optional assignee user UUIDs (milestones support "
                            "multiple assignees)."
                        ),
                    },
                },
                "required": ["title"],
            },
            "existing_milestone_id": {
                "type": "INTEGER",
                "description": (
                    "Optional: attach all top-level tasks to this EXISTING "
                    "milestone instead of creating one (resolve via "
                    "list_milestones)."
                ),
            },
            "parent_task_id": {
                "type": "INTEGER",
                "description": (
                    "Optional: create every top-level task in the batch as a "
                    "SUB-TASK of this existing task ('break this task down "
                    "into sub-tasks'). They inherit its milestone and "
                    "sprint. Mutually exclusive with milestone / "
                    "existing_milestone_id. Use fetch_task first to read "
                    "the task and its comments."
                ),
            },
            "tasks": {
                "type": "ARRAY",
                "items": _TASK_ITEM_SCHEMA,
                "description": (
                    f"1-{_MAX_PLAN_TASKS} tasks, parents ordered before their "
                    "sub-tasks."
                ),
            },
        },
        "required": ["project_id", "tasks"],
    },
    run=_run,
    requires_approval=True,
)
