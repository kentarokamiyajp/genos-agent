"""`list_tasks` tool — structured task query.

Complements `search_knowledge_base` (semantic) with a precise ORM-backed
filter for structural questions: "what are my overdue tasks?", "list all
WIP tasks in project X", "which tasks are assigned to me?".

ACL contract (defence-in-depth):
  * Base scope: `team_id = ctx.team_id` — never crosses tenant boundary.
  * Visibility scope: the query is further restricted to tasks that the
    requesting user legitimately sees:
      - Tasks in projects where ctx.user_id is a ProjectMember, OR
      - Tasks where ctx.user_id is the assignee, OR
      - Tasks where ctx.user_id is the reporter.
    This mirrors the `task_acl_user_ids` derivation used by fetch_task but
    applied as a Django Q-filter so the whole result set is scoped in one
    query rather than per-row.
  * If the caller additionally filters by `project_id`, we verify the user
    is a member of that specific project before narrowing the queryset.
    This catches the case where a user is assignee on a task in a project
    they aren't a member of — they shouldn't be able to enumerate all
    tasks in that project just because they appear on one.

All user ids used for ACL come from `ctx` (server-trusted), never from the
LLM's function-call arguments.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Q
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneMaster
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_LIMIT = 50
_VALID_STATUSES = {"Open", "WIP", "Pending", "Closed"}
# Mirror of the canonical frontend enum in `taskMeta.ts` (Minimal/Low/
# Normal/High/Critical). Invalid values are rejected with a clear error
# so the model corrects course rather than silently returning [].
_VALID_PRIORITIES = {"Minimal", "Low", "Normal", "High", "Critical"}


def _milestone_task_q(milestone: MilestoneMaster) -> Q:
    """Predicate for "tasks belonging to this milestone".

    Matches `_serialize_milestone` in milestone_views.py:200-208 verbatim
    so agent rollups reconcile with the UI:
      * tasks with a direct FK to this milestone, OR
      * tasks whose `parent_task_id` is the milestone's backing task
        (the table renders these as subtasks).
    Preserves the existing quirk where the FK branch does NOT exclude
    `is_milestone=True` — the divergence would otherwise break the
    reconciliation contract.
    """
    q = Q(milestone=milestone, is_deleted=False, is_init_task=False)
    if milestone.task_id is not None:
        q = q | Q(
            parent_task_id=milestone.task_id,
            is_deleted=False,
            is_init_task=False,
            is_milestone=False,
        )
    return q


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # --- Derive the set of project ids the user belongs to. ---
    # Used both for the base ACL filter and for validating explicit
    # project_id requests.
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    # --- Base queryset: tenant-scoped, soft-delete excluded. ---
    qs = TaskMaster.objects.filter(
        team_id=ctx.team_id,
        is_deleted=False,
        is_init_task=False,
    ).filter(
        # ACL row-filter: mirrors task_acl_user_ids logic as a set-based
        # predicate so we get a single query rather than per-row checks.
        Q(project_id__in=member_project_ids)
        | Q(assignee_id=ctx.user_id)
        | Q(reporter_id=ctx.user_id)
    )

    # --- Optional caller filters ---

    raw_project_id = args.get("project_id")
    if raw_project_id is not None:
        try:
            project_id = int(raw_project_id)
        except (TypeError, ValueError):
            raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")
        # Extra membership check: enumerating all tasks in a project the
        # user isn't a member of is not allowed even if they're the
        # assignee on some tasks within it.
        if project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to list all tasks in project {project_id}. "
                "You are not a member of that project."
            )
        qs = qs.filter(project_id=project_id)

    raw_statuses = args.get("status")
    if raw_statuses is not None:
        if isinstance(raw_statuses, str):
            raw_statuses = [raw_statuses]
        invalid = set(raw_statuses) - _VALID_STATUSES
        if invalid:
            raise ToolError(
                f"Invalid status value(s): {sorted(invalid)}. "
                f"Must be one of {sorted(_VALID_STATUSES)}."
            )
        qs = qs.filter(status__in=raw_statuses)

    raw_assignee_id = args.get("assignee_id")
    if raw_assignee_id:
        # The caller supplies a user_id (UUID string) they got from
        # get_team_members or get_current_user.  We don't ACL-check the
        # assignee_id itself — it's a filter value, not an auth claim.
        qs = qs.filter(assignee_id=raw_assignee_id)

    raw_priorities = args.get("priority")
    if raw_priorities is not None:
        if isinstance(raw_priorities, str):
            raw_priorities = [raw_priorities]
        invalid_pri = set(raw_priorities) - _VALID_PRIORITIES
        if invalid_pri:
            raise ToolError(
                f"Invalid priority value(s): {sorted(invalid_pri)}. "
                f"Must be one of {sorted(_VALID_PRIORITIES)}."
            )
        qs = qs.filter(priority__in=raw_priorities)

    raw_milestone_id = args.get("milestone_id")
    if raw_milestone_id is not None:
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
                f"Not authorized to list tasks in milestone {milestone_id}. "
                "You are not a member of that project."
            )
        # Use the same Q-union as the milestone UI rollup so subtasks are
        # included. `.distinct()` because the union can match a row twice.
        qs = qs.filter(_milestone_task_q(milestone)).distinct()

    if args.get("overdue_only"):
        today = timezone.now().date()
        qs = qs.exclude(status__in=["Closed", "Deleted"]).filter(
            due_date__isnull=False, due_date__lt=today
        )

    try:
        limit = int(args.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, _MAX_LIMIT))

    qs = qs.select_related("project").order_by("-ts_updated_at")[:limit]

    tasks = []
    for t in qs:
        tasks.append(
            {
                "task_id": t.task_id,
                # Human-readable PRJ-123 form. Used in chip labels and
                # whenever the prose references a task — never expose the
                # raw integer task_id to end users.
                "display_id": t.display_id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "assignee_id": str(t.assignee_id) if t.assignee_id else None,
                "project_id": t.project_id,
                "project_name": t.project.project_name if t.project else None,
            }
        )

    return {
        "tasks": tasks,
        "__summary__": f"Found {len(tasks)} task(s)",
    }


LIST_TASKS = Tool(
    name="list_tasks",
    description=(
        "Structured query for tasks: filter by project, milestone, status, "
        "priority, assignee, or overdue date. Use this instead of "
        "search_knowledge_base when the user asks a structural question like "
        "'what are my open tasks?', 'which Critical tasks are in milestone X?', "
        "'overdue tasks in project Y', or 'list all WIP tasks assigned to me'. "
        "Returns task_id, title, status, priority, due_date, assignee_id, "
        "project_id, and project_name. Prefer naming projects by their "
        "project_name in prose (e.g. 'In **Website Redesign**: ...') rather "
        "than as 'Project N'. Results are scoped to tasks the current user is "
        "authorised to see (project member, assignee, or reporter)."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to one project. Resolve the name to an id with "
                    "`list_projects` first if needed. Omit to search across "
                    "all accessible projects."
                ),
            },
            "milestone_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict to one milestone. Includes both direct-FK tasks "
                    "AND subtasks (tasks whose `parent_task_id` is the "
                    "milestone's backing task). Resolve a milestone name to "
                    "an id with `list_milestones` first. The milestone's "
                    "project membership is checked just like `project_id`."
                ),
            },
            "status": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": ["Open", "WIP", "Pending", "Closed"],
                },
                "description": (
                    "Filter by one or more statuses. Omit to include all " "non-deleted tasks."
                ),
            },
            "priority": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING",
                    "enum": ["Minimal", "Low", "Normal", "High", "Critical"],
                },
                "description": (
                    "Filter by one or more priorities. Useful for 'show me "
                    "Critical tasks under milestone X' or 'High-priority "
                    "open tasks across my projects'."
                ),
            },
            "assignee_id": {
                "type": "STRING",
                "description": (
                    "Filter by assignee UUID. Use get_current_user to get "
                    "the caller's own id, or get_team_members to resolve "
                    "a name to a UUID."
                ),
            },
            "overdue_only": {
                "type": "BOOLEAN",
                "description": (
                    "If true, only return tasks whose due_date is in the past "
                    "and status is not Closed or Deleted."
                ),
            },
            "limit": {
                "type": "INTEGER",
                "description": f"Max results to return (1–{_MAX_LIMIT}). Default 20.",
            },
        },
        "required": [],
    },
    run=_run,
)
