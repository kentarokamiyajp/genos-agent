"""`get_my_throughput` tool — "what I created and closed in the past N days".

The "me" counterpart to `get_task_throughput_stats` (workspace-wide
counts) and `get_top_task_closers` (leaderboard). Aggregates
`TaskActivity` rows where the caller is the actor and the action_type
is `created` or `closed`. Optionally bucketed by day or week.

Why sourced from `TaskActivity` and not `TaskMaster.ts_created_at`:
reopens are common — a task closed today, reopened tomorrow, closed
again next week shows up once on each close in the activity feed, so
the closed count matches what the user actually did. A `ts_updated_at`
filter would either double-count (any edit) or miss reopens.

ACL contract:
  * Tenant guard: `task__team_id == ctx.team_id`.
  * Membership scope: `task__project_id__in=member_project_ids`. The
    actor filter is naturally me-scoped, but pinning to member projects
    too prevents counting activity on a task whose project the user has
    since left.
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
_DEFAULT_DAYS = 7
_VALID_BUCKETS = {"day", "week"}
_RECENTLY_CLOSED_LIMIT = 10


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        days = int(args.get("days", _DEFAULT_DAYS))
    except (TypeError, ValueError):
        raise ToolError(f"`days` must be an integer (got {args.get('days')!r}).")
    if not 1 <= days <= _MAX_DAYS:
        raise ToolError(f"`days` must be between 1 and {_MAX_DAYS} (got {days}).")

    bucket = args.get("bucket")
    if bucket is not None and bucket not in _VALID_BUCKETS:
        raise ToolError(
            f"`bucket` must be one of {sorted(_VALID_BUCKETS)} or omitted (got {bucket!r})."
        )

    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    since = timezone.now() - timedelta(days=days)
    base_qs = TaskActivity.objects.filter(
        ts_created_at__gte=since,
        actor_id=ctx.user_id,
        task__team_id=ctx.team_id,
        task__project_id__in=member_project_ids,
        task__is_deleted=False,
        task__is_init_task=False,
        action_type__in=[
            TaskActivityActionType.CREATED.value,
            TaskActivityActionType.CLOSED.value,
        ],
    )

    created_expr = Count(Case(When(action_type=TaskActivityActionType.CREATED.value, then=1)))
    closed_expr = Count(Case(When(action_type=TaskActivityActionType.CLOSED.value, then=1)))

    agg = base_qs.aggregate(created=created_expr, closed=closed_expr)
    created_total = agg.get("created") or 0
    closed_total = agg.get("closed") or 0
    net = created_total - closed_total

    result: dict[str, Any] = {
        "days": days,
        "created_total": created_total,
        "closed_total": closed_total,
        "net": net,
    }

    if bucket is not None:
        trunc = TruncDay if bucket == "day" else TruncWeek
        rows = (
            base_qs.annotate(bucket_start=trunc("ts_created_at"))
            .values("bucket_start")
            .annotate(created=created_expr, closed=closed_expr)
            .order_by("bucket_start")
        )
        buckets: list[dict[str, Any]] = []
        for row in rows:
            b = row["bucket_start"]
            buckets.append(
                {
                    "bucket_start": b.date().isoformat() if b is not None else None,
                    "created": row.get("created") or 0,
                    "closed": row.get("closed") or 0,
                }
            )
        result["bucket"] = bucket
        result["buckets"] = buckets

    # Recently-closed list — the actual tasks the user has shipped, so
    # the agent can name them ("you closed WRD-3 yesterday").
    recent_closed_rows = (
        base_qs.filter(action_type=TaskActivityActionType.CLOSED.value)
        .select_related("task", "task__project")
        .order_by("-ts_created_at")[:_RECENTLY_CLOSED_LIMIT]
    )
    recently_closed: list[dict[str, Any]] = []
    for r in recent_closed_rows:
        task = r.task
        if task is None:
            continue
        recently_closed.append(
            {
                "task_id": task.task_id,
                "display_id": task.display_id,
                "title": task.title or "",
                "closed_at": r.ts_created_at.isoformat(),
                "project_id": task.project_id,
                "project_name": task.project.project_name if task.project_id else None,
            }
        )
    result["recently_closed"] = recently_closed

    bucket_note = f" ({len(result.get('buckets', []))} non-empty {bucket}(s))" if bucket else ""
    result["__summary__"] = (
        f"Past {days} day(s): {created_total} created, {closed_total} closed "
        f"(net {net:+d}){bucket_note}"
    )

    return result


GET_MY_THROUGHPUT = Tool(
    name="get_my_throughput",
    description=(
        "How many tasks the current user created and closed in the past "
        "N days, optionally bucketed by day or week. Sourced from the "
        "task activity log (`TaskActivity`) so reopen/close cycles count "
        "correctly. Use for 'what did I close this week?', 'my pace last "
        "month', 'my throughput', or 'what tasks did I ship?'. Returns "
        "created_total, closed_total, net (created − closed), optional "
        "buckets (when bucket=day|week), and recently_closed (up to 10 "
        "of the most recent closes). Bucketed responses omit zero-"
        "activity days/weeks — treat missing buckets as zero."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "days": {
                "type": "INTEGER",
                "description": (
                    f"Window length in days (1–{_MAX_DAYS}). Default "
                    f"{_DEFAULT_DAYS} (one week)."
                ),
            },
            "bucket": {
                "type": "STRING",
                "enum": ["day", "week"],
                "description": (
                    "Optional. Group counts by day or week. Omit to "
                    "return a single window-wide total."
                ),
            },
        },
        "required": [],
    },
    run=_run,
)
