"""`update_tasks_bulk` composite write tool — organize many tasks in ONE approval.

The tool behind "reprioritize / organize / clean up the tasks in this
milestone": the agent reads current state (`list_tasks` +
`list_task_dependencies`), then proposes per-task priority / due-date /
status / effort changes — each carrying a one-sentence `rationale` the
user reads in the approval card — and ONE approval applies all of them.
Without it, a 10-task organize pass is ten `update_task` proposals and
ten Approve clicks.

Semantics: **validate-all-then-apply-all, atomic.** The user approved a
specific plan; a partially-applied one is worse than a clean failure
the model can re-propose. Phase 1 resolves every task in one query and
checks existence / tenant / ACL / enums / dates — any problem fails the
whole batch with every offending `updates[i]` named. Phase 2 applies
the field diffs (`update_task`'s skip-no-op semantics) inside one
transaction. State can drift between the pause and the approval —
that's why validation runs at approve time, not proposal time.

`rationale` is never persisted onto task rows — it lives in the run's
`arguments_json` (audit trail) and the approval preview only.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from django.db import transaction

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.task_models import TaskMaster
from origin.search_engine.agent.tools.base import Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.task_enums import (
    EFFORT_ENUM,
    PRIORITY_ENUM,
    STATUS_ENUM,
    VALID_EFFORTS,
    VALID_PRIORITIES,
    VALID_STATUSES,
)
from origin.services.task_cache import invalidate_project_tasks_cache

_MAX_BULK_UPDATES = 30
# Fields an update row may change, in apply order.
_FIELD_SPECS = (
    ("priority", VALID_PRIORITIES),
    ("effort_level", VALID_EFFORTS),
    ("status", VALID_STATUSES),
)


def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    updates = args.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ToolError("`updates` must be a non-empty array.")
    if len(updates) > _MAX_BULK_UPDATES:
        raise ToolError(
            f"Too many updates: {len(updates)} (max {_MAX_BULK_UPDATES}). "
            "Split the batch and tell the user."
        )

    # ---- phase 1: resolve + validate EVERYTHING before any write ----
    errors: list[str] = []
    task_ids: list[int] = []
    seen_ids: set[int] = set()
    for i, row in enumerate(updates):
        where = f"updates[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: must be an object.")
            continue
        try:
            tid = int(row.get("task_id"))
        except (TypeError, ValueError):
            errors.append(f"{where}: `task_id` must be an integer (got {row.get('task_id')!r}).")
            continue
        if tid in seen_ids:
            errors.append(f"{where}: duplicate task_id {tid}.")
            continue
        seen_ids.add(tid)
        task_ids.append(tid)

    tasks_by_id = {
        t.task_id: t
        for t in TaskMaster.objects.select_related("project").filter(task_id__in=task_ids)
    }

    # Set-based ACL, mirroring `list_tasks`: one membership query for the
    # requester instead of a per-task `task_acl_user_ids` call.
    member_project_ids = set(
        ProjectMembers.objects.filter(
            attendee_id=ctx.user_id,
            project__team_id=ctx.team_id,
            project__is_deleted=False,
        ).values_list("project_id", flat=True)
    )

    def _may_edit(task: TaskMaster) -> bool:
        return (
            task.project_id in member_project_ids
            or str(task.assignee_id or "") == ctx.user_id
            or str(task.reporter_id or "") == ctx.user_id
        )

    planned: list[tuple[TaskMaster, dict[str, Any], str]] = []
    for i, row in enumerate(updates):
        if not isinstance(row, dict):
            continue
        where = f"updates[{i}]"
        try:
            tid = int(row.get("task_id"))
        except (TypeError, ValueError):
            continue
        task = tasks_by_id.get(tid)
        if task is None:
            errors.append(f"{where}: task {tid} not found.")
            continue
        if task.is_deleted:
            errors.append(f"{where}: task {tid} has been deleted.")
            continue
        if str(getattr(task, "team_id", "") or "") != ctx.team_id:
            errors.append(f"{where}: not authorized — task {tid} is in a different team.")
            continue
        if not _may_edit(task):
            errors.append(f"{where}: not authorized to update task {tid}.")
            continue

        changes: dict[str, Any] = {}
        for field, valid in _FIELD_SPECS:
            value = row.get(field)
            if value is None:
                continue
            if value not in valid:
                errors.append(
                    f"{where}: `{field}` must be one of {sorted(valid)} (got {value!r})."
                )
                continue
            changes[field] = value

        if "due_date" in row and row["due_date"] is not None:
            raw_due = row["due_date"]
            if raw_due == "":
                changes["due_date"] = None
            else:
                try:
                    changes["due_date"] = date.fromisoformat(str(raw_due))
                except (TypeError, ValueError):
                    errors.append(
                        f"{where}: `due_date` must be an ISO 8601 date "
                        f"(got {raw_due!r}). Use '' to clear."
                    )

        rationale = (row.get("rationale") or "").strip()
        if not rationale:
            errors.append(f"{where}: `rationale` is required — say WHY in one sentence.")
        if not changes:
            errors.append(
                f"{where}: no changes given for task {tid} — provide at least one of "
                "priority / effort_level / status / due_date."
            )
        planned.append((task, changes, rationale))

    if errors:
        raise ToolError(
            "Bulk update validation failed (nothing was applied):\n- " + "\n- ".join(errors[:25])
        )

    # ---- phase 2: apply all diffs atomically ----
    updated: list[dict[str, Any]] = []
    noops: list[int] = []
    try:
        with transaction.atomic():
            for task, changes, _rationale in planned:
                update_fields: list[str] = []
                changed: list[str] = []
                for field, new_value in changes.items():
                    if getattr(task, field) != new_value:
                        setattr(task, field, new_value)
                        update_fields.append(field)
                        changed.append(
                            f"{field}(cleared)" if field == "due_date" and new_value is None
                            else field
                        )
                if not update_fields:
                    noops.append(task.task_id)
                    continue
                task.save(update_fields=update_fields)
                updated.append(
                    {
                        "task_id": task.task_id,
                        "display_id": task.display_id,
                        "title": task.title,
                        "project_id": task.project_id,
                        "changed_fields": changed,
                    }
                )
    except Exception as e:  # noqa: BLE001 — surface as ToolError for the model
        raise ToolError(f"Bulk update failed (nothing was applied): {e}")

    # Priorities/statuses feed the project table's ordering — drop each
    # touched project's cached listing after commit.
    for pid in {row["project_id"] for row in updated}:
        if pid is not None:
            invalidate_project_tasks_cache(ctx.team_id, pid)

    field_counts: dict[str, int] = {}
    for row in updated:
        for f in row["changed_fields"]:
            key = f.removesuffix("(cleared)")
            field_counts[key] = field_counts.get(key, 0) + 1
    parts = [f"{n} {field}" for field, n in sorted(field_counts.items())] or ["no-op"]
    summary = f"Updated {len(updated)} task(s): " + ", ".join(parts)
    if noops:
        summary += f" ({len(noops)} already up to date)"

    return {
        "updated": updated,
        "noops": noops,
        "__summary__": summary,
    }


UPDATE_TASKS_BULK = Tool(
    name="update_tasks_bulk",
    description=(
        "Change priority / due_date / status / effort_level on MANY tasks in "
        "ONE approval — the tool for 'reprioritize', 'organize', or 'clean "
        "up' the tasks in a milestone or project. REQUIRES USER APPROVAL — "
        "the user sees every proposed change WITH your per-task rationale "
        "and approves once. ALWAYS read current state first (list_tasks, "
        "and list_task_dependencies for blocker ordering) so every change "
        "is a real, justified change. NEVER call update_task repeatedly "
        "for a batch; update_task remains correct for ONE task. All-or-"
        "nothing: one invalid entry rejects the whole batch. Cannot set "
        "status to Deleted."
    ),
    parameters_schema={
        "type": "OBJECT",
        "properties": {
            "updates": {
                "type": "ARRAY",
                "description": (
                    f"1-{_MAX_BULK_UPDATES} per-task changes. Include ONLY "
                    "tasks that actually change, and only the fields that "
                    "change."
                ),
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "task_id": {
                            "type": "INTEGER",
                            "description": "Numeric task id (from list_tasks).",
                        },
                        "priority": {
                            "type": "STRING",
                            "enum": PRIORITY_ENUM,
                            "description": "New priority. Omit to leave unchanged.",
                        },
                        "due_date": {
                            "type": "STRING",
                            "description": (
                                "New ISO 8601 date (YYYY-MM-DD). Omit to leave "
                                "unchanged. Pass '' to clear."
                            ),
                        },
                        "status": {
                            "type": "STRING",
                            "enum": STATUS_ENUM,
                            "description": "New status. Omit to leave unchanged.",
                        },
                        "effort_level": {
                            "type": "STRING",
                            "enum": EFFORT_ENUM,
                            "description": "New effort estimate. Omit to leave unchanged.",
                        },
                        "rationale": {
                            "type": "STRING",
                            "description": (
                                "One sentence: WHY this change (the user reads "
                                "this in the approval card)."
                            ),
                        },
                    },
                    "required": ["task_id", "rationale"],
                },
            },
        },
        "required": ["updates"],
    },
    run=_run,
    requires_approval=True,
)
