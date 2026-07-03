"""`get_my_focus_tasks` tool — prioritised "what to look at next".

Returns the caller's most-important active tasks, sorted with the same
comparator the dashboard's "Up Next" card uses
(`genos-frontend/src/features/tasks/components/dashboard/TaskHomeContent.tsx`
lines 523-558). Sorting tiers, in order:

  1. Overdue float to top — tasks past due_date come first; among the
     overdue group, more-overdue first (earliest due_date wins).
  2. Priority rank — Critical(0) → High(1) → Normal(2) → Low(3) →
     Minimal(4) → none(5).
  3. Soonest due_date — no due_date sorts last (Infinity).
  4. Most recently updated — newest `ts_updated_at` wins the final tie.

The Python comparator below mirrors that comparator chain verbatim;
matching it exactly is the contract — diverging would make the agent
say "look at A first" while the dashboard says "look at B first".

ACL contract:
  * Tenant guard: ctx.team_id.
  * Scope: `assignee_id == ctx.user_id` AND the task's project is in
    ctx.user_id's `ProjectMembers` set.
  * `status NOT IN ("Closed", "Deleted")` — focus is on actionable work.
"""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any

from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_LIMIT = 20
_DEFAULT_LIMIT = 5
_PRIORITY_RANK = {
    "Critical": 0,
    "High": 1,
    "Normal": 2,
    "Low": 3,
    "Minimal": 4,
}
_INACTIVE_STATUSES = ["Closed", "Deleted"]


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        raise ToolError(f"`limit` must be an integer (got {args.get('limit')!r}).")
    limit = max(1, min(limit, _MAX_LIMIT))

    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    today = timezone.now().date()

    # Fetch a bounded pool, then sort in Python with the exact comparator
    # chain the dashboard uses. We over-fetch (limit * 5, capped at 100)
    # so a project with hundreds of tasks doesn't push the lowest-priority
    # rows out of view before the sort runs.
    pool_cap = min(max(limit * 5, 50), 100)
    qs = (
        TaskMaster.objects.filter(
            team_id=ctx.team_id,
            assignee_id=ctx.user_id,
            project_id__in=member_project_ids,
            is_deleted=False,
            is_init_task=False,
        )
        .exclude(status__in=_INACTIVE_STATUSES)
        .select_related("project")
        .order_by("-ts_updated_at")[:pool_cap]
    )
    rows = list(qs)

    none_rank = max(_PRIORITY_RANK.values()) + 1

    def _compare(a: TaskMaster, b: TaskMaster) -> int:
        # Stage 1 — overdue first; among overdue rows, earliest due wins.
        a_overdue = a.due_date is not None and a.due_date < today
        b_overdue = b.due_date is not None and b.due_date < today
        if a_overdue != b_overdue:
            return -1 if a_overdue else 1
        if a_overdue and b_overdue:
            # both overdue — earlier due_date comes first
            if a.due_date != b.due_date:
                return -1 if a.due_date < b.due_date else 1
        # Stage 2 — priority rank (Critical=0 → Minimal=4 → none=5)
        pa = _PRIORITY_RANK.get(a.priority or "", none_rank)
        pb = _PRIORITY_RANK.get(b.priority or "", none_rank)
        if pa != pb:
            return -1 if pa < pb else 1
        # Stage 3 — soonest due_date, with None sorting last.
        if a.due_date is None and b.due_date is None:
            pass
        elif a.due_date is None:
            return 1
        elif b.due_date is None:
            return -1
        elif a.due_date != b.due_date:
            return -1 if a.due_date < b.due_date else 1
        # Stage 4 — most recently updated wins the final tie.
        au = a.ts_updated_at
        bu = b.ts_updated_at
        if au != bu:
            return -1 if au > bu else 1
        return 0

    rows.sort(key=cmp_to_key(_compare))
    rows = rows[:limit]

    tasks: list[dict[str, Any]] = []
    for t in rows:
        is_overdue = t.due_date is not None and t.due_date < today
        days_until_due: int | None = None
        if t.due_date is not None:
            days_until_due = (t.due_date - today).days
        tasks.append(
            {
                "task_id": t.task_id,
                "display_id": t.display_id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "is_overdue": is_overdue,
                "days_until_due": days_until_due,
                "project_id": t.project_id,
                "project_name": t.project.project_name if t.project else None,
            }
        )

    if tasks:
        head = ", ".join(f"{t['display_id']} {t['title']}" for t in tasks[:3])
        summary = f"Top {len(tasks)} to focus on: {head}" + (" …" if len(tasks) > 3 else "")
    else:
        summary = "No active tasks assigned to you."

    return {
        "tasks": tasks,
        "__summary__": summary,
    }


GET_MY_FOCUS_TASKS = Tool(
    name="get_my_focus_tasks",
    description=(
        "Prioritised list of the current user's active tasks, ranked by "
        "the same comparator the dashboard's 'Up Next' card uses: "
        "overdue first → priority (Critical → High → Normal → Low → "
        "Minimal → none) → soonest due_date → most recently updated. "
        "Use for 'what should I work on first?', 'today's priorities', "
        "'what's most important right now?'. Returns up to `limit` "
        "tasks (default 5), each with is_overdue, days_until_due, and "
        "project info. For aggregate counts instead, use "
        "`get_my_task_summary`."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "limit": {
                "type": "INTEGER",
                "description": (
                    f"Max tasks to return (1–{_MAX_LIMIT}). Default "
                    f"{_DEFAULT_LIMIT} — matches the dashboard's top-5 "
                    "view."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
