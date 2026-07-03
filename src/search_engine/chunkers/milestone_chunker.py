"""Milestone chunker.

One chunk per milestone holding its title + description, so milestones
are searchable by title and body — mirroring the task chunker.

A milestone is conceptually a task with `is_milestone=True`; the backing
`TaskMaster` row (`MilestoneMaster.task`) owns its body/comments. We
surface that backing `task_id` on the chunk so a Spotlight result can
deep-link to the milestone via the same
`/workspace/tasks/project/<project>/task/<task>` view the rest of the app
already uses to open milestones — no new route needed.

ACL = project members of the milestone's project, plus its multi-
assignees (`MilestoneAssignees`) and its reporter, so a user can find
milestones they're on even outside project membership (same rule as the
task chunker's assignee/reporter widening).

Skipped: `is_deleted=True` rows, and milestones with no team or no
title/description text to index.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator, Optional

from origin.models.project.prj_models import ProjectMembers
from origin.models.task.milestone_models import MilestoneAssignees, MilestoneMaster
from origin.search_engine.chunkers.base import (
    Chunk,
    EntityChunks,
    iso,
    make_snippet,
)
from origin.search_engine.text_extraction import extract_text


def iter_milestone_chunks(since: Optional[datetime] = None) -> Iterator[EntityChunks]:
    """Yield one EntityChunks (single chunk) per milestone."""
    qs = MilestoneMaster.objects.filter(is_deleted=False).select_related("team", "project")
    if since is not None:
        qs = qs.filter(ts_updated_at__gte=since)
    milestones = list(qs)
    if not milestones:
        return

    # Pre-load project-member ACLs.
    project_ids = {m.project_id for m in milestones if m.project_id}
    members_by_project: dict[int, list[str]] = defaultdict(list)
    for row in ProjectMembers.objects.filter(project_id__in=project_ids).values(
        "project_id", "attendee_id"
    ):
        if row["attendee_id"] is not None:
            members_by_project[row["project_id"]].append(str(row["attendee_id"]))

    # Pre-load multi-assignees per milestone.
    milestone_ids = [m.milestone_id for m in milestones]
    assignees_by_ms: dict[int, list[str]] = defaultdict(list)
    for row in MilestoneAssignees.objects.filter(milestone_id__in=milestone_ids).values(
        "milestone_id", "user_id"
    ):
        if row["user_id"] is not None:
            assignees_by_ms[row["milestone_id"]].append(str(row["user_id"]))

    for m in milestones:
        if not m.team_id:
            continue
        team_id = str(m.team_id)
        project_id = str(m.project_id) if m.project_id else None

        acl = set(members_by_project.get(m.project_id, [])) if m.project_id else set()
        acl.update(assignees_by_ms.get(m.milestone_id, []))
        if m.reporter_id:
            acl.add(str(m.reporter_id))

        title_clean = (m.title or "").strip()
        body_text = extract_text(m.description)
        search_text = "\n".join(p for p in [title_clean, body_text] if p).strip()
        # Nothing to index (no title and no body).
        if not search_text:
            continue

        entity_id = f"milestone:{m.milestone_id}"
        chunk = Chunk(
            chunk_id=f"milestone:{m.milestone_id}:title_content",
            entity_type="milestone",
            entity_id=entity_id,
            chunk_type="milestone_title_content",
            team_id=team_id,
            acl_user_ids=sorted(acl),
            title=m.title or f"Milestone {m.milestone_id}",
            search_text=search_text,
            snippet_text=make_snippet(search_text),
            project_id=project_id,
            # Backing task — lets Spotlight open the milestone via the
            # existing task deep-link. None for legacy milestones whose
            # backing task hasn't been auto-created yet.
            task_id=str(m.task_id) if m.task_id else None,
            created_at=iso(m.ts_created_at),
            updated_at=iso(m.ts_updated_at),
        )
        yield EntityChunks(entity_type="milestone", entity_id=entity_id, chunks=[chunk])
