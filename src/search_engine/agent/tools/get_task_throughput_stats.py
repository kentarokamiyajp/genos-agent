"""`get_task_throughput_stats` tool — created vs. closed task counts.

Answers "how many tasks were created/closed in the past N days?" with an
optional day/week breakdown. Backed by `TaskActivity` rows
(action_type='created' / 'closed') rather than `TaskMaster.ts_created_at`
so reopen/close cycles are counted correctly and the closed timeline
matches what users actually saw in the activity feed.

ACL contract:
  * Tenant guard: scoped to ctx.team_id via `task__team_id`.
  * Membership guard: counts only span projects where ctx.user_id is a
    ProjectMember. Aggregating over the wider "assignee/reporter"
    visibility set would leak activity volume from projects the user
    isn't a member of, even if they touched one task inside.
  * Explicit `project_id`: validated against the user's membership set.

Sparse-bucket note: zero-activity days/weeks are dropped from the
`buckets` array (group-by yields no row for empty groups). The tool
description tells the model to treat missing buckets as zeros.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db.models import Case, Count, When
from django.db.models.functions import TruncDay, TruncWeek
from django.utils import timezone

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_activity_models import TaskActivity, TaskActivityActionType
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError

_MAX_DAYS = 365
_VALID_BUCKETS = {"day", "week"}


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # --- Validate `days`. ---
    raw_days = args.get("days")
    try:
        days = int(raw_days)
    except (TypeError, ValueError):
        raise ToolError(f"`days` must be an integer (got {raw_days!r}).")
    if not 1 <= days <= _MAX_DAYS:
        raise ToolError(f"`days` must be between 1 and {_MAX_DAYS} (got {days}).")

    # --- Validate `bucket`. ---
    bucket = args.get("bucket")
    if bucket is not None and bucket not in _VALID_BUCKETS:
        raise ToolError(
            f"`bucket` must be one of {sorted(_VALID_BUCKETS)} or omitted (got {bucket!r})."
        )

    # --- Derive the user's project membership set (ACL scope). ---
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    # --- Optional explicit project_id, validated against membership. ---
    raw_project_id = args.get("project_id")
    scoped_project_id: int | None = None
    if raw_project_id is not None:
        try:
            scoped_project_id = int(raw_project_id)
        except (TypeError, ValueError):
            raise ToolError(f"`project_id` must be an integer (got {raw_project_id!r}).")
        if scoped_project_id not in member_project_ids:
            raise ToolError(
                f"Not authorized to query project {scoped_project_id}. "
                "You are not a member of that project."
            )
        scoped_project_ids = {scoped_project_id}
    else:
        scoped_project_ids = member_project_ids

    # --- Base queryset. Filter through `task__` so on_delete=SET_NULL
    #     orphan rows on TaskActivity.team/.project can't slip past. ---
    since = timezone.now() - timedelta(days=days)
    qs = TaskActivity.objects.filter(
        ts_created_at__gte=since,
        task__team_id=ctx.team_id,
        task__project_id__in=scoped_project_ids,
        task__is_deleted=False,
        task__is_init_task=False,
        action_type__in=[
            TaskActivityActionType.CREATED.value,
            TaskActivityActionType.CLOSED.value,
        ],
    )

    created_expr = Count(Case(When(action_type=TaskActivityActionType.CREATED.value, then=1)))
    closed_expr = Count(Case(When(action_type=TaskActivityActionType.CLOSED.value, then=1)))

    if bucket is None:
        agg = qs.aggregate(created=created_expr, closed=closed_expr)
        created_total = agg.get("created") or 0
        closed_total = agg.get("closed") or 0
        net = created_total - closed_total
        return {
            "days": days,
            "project_id": scoped_project_id,
            "created_total": created_total,
            "closed_total": closed_total,
            "__summary__": (
                f"Past {days} day(s): {created_total} created, "
                f"{closed_total} closed (net {net:+d})"
            ),
        }

    trunc = TruncDay if bucket == "day" else TruncWeek
    rows = (
        qs.annotate(bucket_start=trunc("ts_created_at"))
        .values("bucket_start")
        .annotate(created=created_expr, closed=closed_expr)
        .order_by("bucket_start")
    )

    buckets: list[dict[str, Any]] = []
    created_total = 0
    closed_total = 0
    for row in rows:
        b = row["bucket_start"]
        c_cnt = row.get("created") or 0
        x_cnt = row.get("closed") or 0
        created_total += c_cnt
        closed_total += x_cnt
        buckets.append(
            {
                "bucket_start": b.date().isoformat() if b is not None else None,
                "created": c_cnt,
                "closed": x_cnt,
            }
        )

    return {
        "days": days,
        "project_id": scoped_project_id,
        "bucket": bucket,
        "buckets": buckets,
        "created_total": created_total,
        "closed_total": closed_total,
        "__summary__": (
            f"Past {days} day(s) by {bucket}: {created_total} created, "
            f"{closed_total} closed across {len(buckets)} non-empty {bucket}(s)"
        ),
    }


GET_TASK_THROUGHPUT_STATS = Tool(
    name="get_task_throughput_stats",
    description=(
        "Aggregate count of tasks created and tasks closed within the past "
        "N days, optionally bucketed by day or week. Sourced from the task "
        "activity log so reopen/close cycles count correctly. Use for "
        "'how many tasks were closed last week?', 'task throughput for "
        "the past month', or 'show me the weekly burn-down'. Counts are "
        "scoped to projects the current user is a member of; pass "
        "`project_id` to narrow to one project (membership enforced). "
        "Bucketed responses omit zero-activity days/weeks — treat missing "
        "buckets as zero."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "days": {
                "type": "INTEGER",
                "description": f"Window length in days (1–{_MAX_DAYS}).",
            },
            "project_id": {
                "type": "INTEGER",
                "description": (
                    "Restrict counts to one project. Resolve the name to an "
                    "id with `list_projects` first if needed. Omit to count "
                    "across all accessible projects."
                ),
            },
            "bucket": {
                "type": "STRING",
                "enum": ["day", "week"],
                "description": (
                    "Optional. Group counts by day or week. Omit to return a "
                    "single window-wide total."
                ),
            },
        },
        "required": ["days"],
    },
    run=_run,
)
