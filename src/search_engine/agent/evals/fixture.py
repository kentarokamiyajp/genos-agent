"""Deterministic eval fixture.

Reseeds a known team's worth of demo data (via `demo_seeder`) under
fixed UUIDs so retrieval / agent cases can assert against stable
content (titles, statuses, project membership) without depending on a
hand-managed dev DB.

Usage:

    python manage.py agent_eval_setup            # ensure-or-reseed
    python manage.py agent_eval_setup --reseed   # tear down + reseed
    python manage.py agent_eval --retrieval      # then run the suite

Notes
-----
* Fixture UUIDs are all zero-prefixed; they cannot collide with prod
  users / teams (which are random uuid4).
* The seeded team carries `is_demo=True`, so the existing
  `delete_demo_team_data` helper cleans it up if you ever need to.
* Eval cases reference entities by **title substring**, not by
  numeric id — see `runner.py::must_contain_title_in_top_n`. This
  keeps the YAML stable across reseedings (where auto-incremented ids
  shift).
"""

from __future__ import annotations

import logging
import uuid

from django.core.management import call_command
from django.db import transaction

from origin.models.common.team_models import TeamMaster
from origin.models.common.user_models import CustomUser
from origin.models.note.common_note_models import NotePermissionMaster
from origin.models.note.personal_note_models import PersonalNoteMaster
from origin.services.demo_seeder import (
    create_demo_environment,
    delete_demo_team_data,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multilingual fixture content (v3 — multilingual BM25)
# ---------------------------------------------------------------------------
# A few non-English personal notes layered on top of the English demo
# data so retrieval cases can exercise the ICU + kuromoji analyzers
# end-to-end (see retrieval_cases.yaml section "MULTI"). We do NOT
# localise the 3267-line demo_seeder — this is a small, dedicated set.
# Each note carries a distinctive proper-noun / token in its TITLE so
# `must_contain_title_in_top_n` assertions stay stable across reseeds.


def _ml_block(text: str, *, heading: bool = False) -> dict:
    """Minimal BlockNote block (paragraph or level-3 heading)."""
    return {
        "id": uuid.uuid4().hex,
        "type": "heading" if heading else "paragraph",
        "props": {"level": 3} if heading else {},
        "content": [{"text": text, "type": "text", "styles": {}}],
        "children": [],
    }


# (title, [body lines]) — body[0] is rendered as a heading, rest as paragraphs.
_MULTILINGUAL_NOTES: list[tuple[str, list[str]]] = [
    # Japanese: kuromoji target. Title carries 機械学習 (exact keyword);
    # body carries 走った (inflected — a 走る query matches via
    # kuromoji_baseform) and 自然言語処理.
    (
        "機械学習チームの東京マラソン参加記録",
        [
            "週末の予定",
            "先週末、機械学習チームのメンバーと一緒に東京マラソンに参加して42キロを走った。",
            "来月は自然言語処理モデルの再学習を完了する予定です。",
        ],
    ),
    # Chinese: ICU segmentation target. Title + body carry 自然语言处理
    # and 搜索引擎 (no spaces — standard analyzer would unigram these).
    (
        "北京团队的自然语言处理与搜索引擎项目",
        [
            "项目概述",
            "我们正在优化搜索引擎的检索性能，并改进自然语言处理模型的准确率。",
            "下个季度将完成机器学习流水线的迁移。",
        ],
    ),
    # Spanish: icu_folding target. Title carries the accented "café";
    # an un-accented "cafe" query matches only via the folded .icu field.
    (
        "Plan de marketing para la página de café",
        [
            "Resumen",
            "Estamos rediseñando la página de café con nuevas reseñas de clientes.",
            "El objetivo es mejorar la conversión en la página de inicio.",
        ],
    ),
    # Arabic: space-delimited but benefits from ICU normalisation.
    # Title carries a distinctive term for the recall assertion.
    (
        "خطة تسويق المقهى الجديد",
        [
            "ملخص",
            "نعمل على إعادة تصميم صفحة المقهى مع مراجعات جديدة من العملاء.",
        ],
    ),
]


def _seed_multilingual_notes(team: TeamMaster, owner: CustomUser) -> list[NotePermissionMaster]:
    """Create the dedicated multilingual personal notes + their owner
    permission rows (the note APIs 403 without an explicit ROLE_OWNER
    row). Returns the permission rows to bulk_create."""
    permissions: list[NotePermissionMaster] = []
    for title, lines in _MULTILINGUAL_NOTES:
        body = [_ml_block(lines[0], heading=True), *[_ml_block(t) for t in lines[1:]]]
        note = PersonalNoteMaster.objects.create(
            team=team,
            owner=owner,
            title=title,
            body=body,
        )
        permissions.append(
            NotePermissionMaster(
                team=team,
                user=owner,
                note_id=note.note_id,
                note_type=1,
                role_id=1,
            )
        )
    return permissions


# Fixed UUIDs used by the eval fixture. Zero-prefixed so they're
# visually distinct from prod uuid4 values and cannot collide.
FIXTURE_USER_ID = uuid.UUID("00000000-0000-4000-8000-00000000ee01")
FIXTURE_USER_EMAIL = "eval-fixture@genos.app"
FIXTURE_USER_USERNAME = "Eval Fixture User"

# Seeder slug that becomes part of project / bot / team display names.
# Keeping it short and constant means re-seedings produce the same
# strings (so cases that match on titles keep matching).
FIXTURE_SHORT = "evalfixt"


def ensure_fixture() -> dict:
    """Create the fixture if it doesn't already exist; return its summary.

    Idempotent: a second call returns the existing team's summary
    without reseeding. Use `reseed_fixture()` to force fresh content.

    Returns:
        {"team_id": str, "user_id": str, "reseeded": bool}
    """
    existing_user = CustomUser.objects.filter(id=FIXTURE_USER_ID).first()
    if existing_user is not None:
        # Find the team they own — the seeder always creates exactly one
        # demo team per call.
        team = TeamMaster.objects.filter(owner=existing_user, is_demo=True).first()
        if team is not None:
            return {
                "team_id": str(team.team_id),
                "user_id": str(FIXTURE_USER_ID),
                "reseeded": False,
            }

    return reseed_fixture()


def reseed_fixture() -> dict:
    """Tear down any existing fixture team and reseed from scratch.

    Use when:
      * the seeder content changed (case assertions need fresh ids)
      * you want a known-clean state for a baseline measurement
      * eval failures point to drifted index data

    The teardown deletes the team's data AND the indexed OpenSearch
    chunks. The reseed runs the full `create_demo_environment` then
    invokes `opensearch_reindex` synchronously so the index is hot by
    the time this returns.
    """
    with transaction.atomic():
        # 1. Clean slate: remove any existing fixture team's data, the
        # fixture user, and the seeded bot peers. The seeder will
        # recreate all three.
        existing_user = CustomUser.objects.filter(id=FIXTURE_USER_ID).first()
        if existing_user is not None:
            for team in TeamMaster.objects.filter(owner=existing_user, is_demo=True):
                delete_demo_team_data(team.team_id)
            existing_user.delete()
        # `delete_demo_team_data` doesn't touch the bot users (they're
        # standalone CustomUser rows). Their emails are deterministic
        # — `demo-bot-{short}-<role>@genos.app` — so we can clean them
        # up by pattern when the fixture slug is fixed.
        CustomUser.objects.filter(email__startswith=f"demo-bot-{FIXTURE_SHORT}-").delete()

        # 2. Recreate the fixture user with a known UUID so cases that
        # need the requesting user can hard-code it. `id` is the PK
        # and isn't assignable after-the-fact, so pass it through the
        # manager's **extra_fields path.
        demo_user = CustomUser.objects.create_user(
            email=FIXTURE_USER_EMAIL,
            username=FIXTURE_USER_USERNAME,
            password=uuid.uuid4().hex,
            id=FIXTURE_USER_ID,
            is_demo=True,
        )

        # 3. Run the seeder with a fixed slug so generated team / bot /
        # project names are byte-identical across reseedings.
        summary = create_demo_environment(demo_user, short=FIXTURE_SHORT)

        # 3b. Layer the dedicated multilingual notes on top (v3 — ICU +
        # kuromoji eval content). Kept out of demo_seeder on purpose.
        team = TeamMaster.objects.get(team_id=summary["team_id"])
        ml_permissions = _seed_multilingual_notes(team, demo_user)
        NotePermissionMaster.objects.bulk_create(ml_permissions)

    # 4. Reindex synchronously so the freshly-written rows are
    # searchable when the eval starts. `since_minutes=60` is generous
    # — the seeder just wrote everything, so timestamps are < 1 minute
    # old, but we pad against clock skew between worker + DB.
    call_command("opensearch_reindex", since_minutes=60)

    return {
        "team_id": summary["team_id"],
        "user_id": str(FIXTURE_USER_ID),
        "reseeded": True,
    }
