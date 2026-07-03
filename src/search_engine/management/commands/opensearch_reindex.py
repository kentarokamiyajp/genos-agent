"""Management command that runs the OpenSearch ingestion pipeline.

Usage:

    # Full reindex (re-evaluates every entity; only re-embeds chunks
    # whose text actually changed).
    python manage.py opensearch_reindex

    # Incremental reindex — only entities updated in the last N minutes.
    # Suitable for a crontab entry.
    python manage.py opensearch_reindex --since-minutes 10

    # Restrict to specific entity types.
    python manage.py opensearch_reindex --entity-types chat task

The command is idempotent: running it twice in a row when nothing has
changed produces zero embedding calls and zero OpenSearch writes (it
still scans Postgres + the RagChunk table).

Schema versions: when the live index was built under a different
`INDEX_SCHEMA_VERSION` than the code expects, the command prints a
warning. The fix is to recreate the index:

    python manage.py opensearch_setup --recreate
    python manage.py opensearch_reindex
"""

import json
from datetime import datetime, timedelta, timezone

from django.core.management.base import CommandError
from opensearchpy.exceptions import NotFoundError

from origin.management.cron_command import CronCommand
from origin.search_engine.index_config import INDEX_SCHEMA_VERSION
from origin.search_engine.ingestion import ingest_all
from origin.search_engine.opensearch_client import get_client, get_index_alias


class Command(CronCommand):
    help = (
        "Re-index chats, tasks, and notes into OpenSearch. By default "
        "runs a full reindex; pass --since-minutes for an incremental "
        "pass (suitable for crontab)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--since-minutes",
            type=int,
            default=None,
            help=(
                "Only re-process entities updated within the last N "
                "minutes. Default: full reindex."
            ),
        )
        parser.add_argument(
            "--since",
            type=str,
            default=None,
            help=("Explicit ISO 8601 timestamp lower-bound. Overrides " "--since-minutes."),
        )
        parser.add_argument(
            "--entity-types",
            nargs="+",
            default=None,
            choices=[
                "chat",
                "task",
                "milestone",
                "note",
                "thread_summary",
                "note_summary",
                "todo",
                "conversation",
                "spotlight_answer",
            ],
            help="Subset of entity types to ingest. Default: all.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Run the chunkers and compute new/changed/stale counts "
                "without calling the embedding API or writing to "
                "OpenSearch / the tracking table."
            ),
        )

    def handle(self, *args, **options):
        # Schema-version sanity check. Best-effort: a mismatched index
        # is still indexable (writes succeed; just the new fields are
        # absent from the live mappings), but the operator should know
        # to recreate so the new keyword fields / subfields work.
        self._warn_on_schema_mismatch()

        # Preflight: fail fast (and before spending any embedding budget)
        # if OpenSearch is unreachable. Otherwise the run would embed the
        # whole corpus, fail every bulk write, and rely on the CronCommand
        # tripwire to catch it only after the wasted work.
        if not options.get("dry_run") and not get_client().ping():
            raise CommandError("OpenSearch is unreachable (ping failed); aborting reindex.")

        since = None
        if options.get("since"):
            since = datetime.fromisoformat(options["since"])
        elif options.get("since_minutes") is not None:
            since = datetime.now(timezone.utc) - timedelta(minutes=options["since_minutes"])

        if since is not None:
            self.stdout.write(f"Incremental reindex since {since.isoformat()}...")
        else:
            self.stdout.write("Full reindex starting...")
        if options.get("dry_run"):
            self.stdout.write("(dry-run: no embeddings, no writes)")

        stats = ingest_all(
            since=since,
            entity_types=options.get("entity_types"),
            dry_run=options.get("dry_run", False),
        )
        self.stdout.write(self.style.SUCCESS("Reindex complete."))
        self.stdout.write(json.dumps(stats.as_dict(), indent=2))

    def _warn_on_schema_mismatch(self):
        """Sample one chunk's `index_schema_version` and compare to the
        code's `INDEX_SCHEMA_VERSION`. A mismatch means the live index
        was built before the current schema; new keyword fields and
        text subfields won't exist on the live mapping, so the new
        chunkers will write fields that get silently dropped.

        Non-fatal. Recovery: `manage.py opensearch_setup --recreate`.
        """
        try:
            client = get_client()
            alias = get_index_alias()
            resp = client.search(
                index=alias,
                body={
                    "size": 1,
                    "_source": ["index_schema_version"],
                    "query": {"match_all": {}},
                },
            )
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                return  # empty index — first reindex, no mismatch to warn about
            live_version = (hits[0].get("_source") or {}).get("index_schema_version")
            if live_version and live_version != INDEX_SCHEMA_VERSION:
                self.stdout.write(
                    self.style.WARNING(
                        f"Index schema mismatch: live index is "
                        f"{live_version!r} but code expects "
                        f"{INDEX_SCHEMA_VERSION!r}. New v2 fields "
                        "(author_id, task_status, .prefix subfield, "
                        "etc.) won't be searchable until the index is "
                        "recreated. Run:\n"
                        "  manage.py opensearch_setup --recreate\n"
                        "  manage.py opensearch_reindex"
                    )
                )
        except (NotFoundError, Exception):  # noqa: BLE001 — never block reindex on a probe failure
            return
