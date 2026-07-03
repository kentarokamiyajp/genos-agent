"""`python manage.py agent_eval_setup` — provision the eval fixture.

Ensures the deterministic eval team / user / seeded data exists and is
indexed in OpenSearch. Idempotent by default; pass `--reseed` to force
a clean teardown + reseed (use when the seeder content has changed).

Run this once before `python manage.py agent_eval --retrieval` (or
`--all`). The fixture team uses fixed UUIDs so eval-case YAML can
hard-code `team_id` / `user_id` without drift.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from origin.search_engine.agent.evals.fixture import (
    FIXTURE_USER_ID,
    ensure_fixture,
    reseed_fixture,
)


class Command(BaseCommand):
    help = "Provision (or re-provision) the deterministic eval fixture."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reseed",
            action="store_true",
            help=(
                "Force a clean teardown + reseed even if the fixture "
                "already exists. Use when seeder content changed."
            ),
        )

    def handle(self, *args, **options):
        force_reseed = options.get("reseed") or False

        if force_reseed:
            self.stdout.write("Reseeding eval fixture from scratch…")
            summary = reseed_fixture()
        else:
            self.stdout.write("Ensuring eval fixture exists…")
            summary = ensure_fixture()

        action = "reseeded" if summary["reseeded"] else "already present"
        self.stdout.write(
            self.style.SUCCESS(
                f"\nEval fixture {action}.\n"
                f"  team_id: {summary['team_id']}\n"
                f"  user_id: {summary['user_id']}\n"
                f"\nYAML cases referencing the fixture should use:\n"
                f'  team_id: "{summary["team_id"]}"\n'
                f'  user_id: "{FIXTURE_USER_ID}"\n'
            )
        )
