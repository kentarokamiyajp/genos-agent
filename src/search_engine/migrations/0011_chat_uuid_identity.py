# Hand-authored UUID chat-identity migration for AgentSession + ThreadSummary
# (chat_id / thread_id IntegerField -> UUIDField). Django was not installed in
# the authoring environment, so this could not be generated or verified there.
# Verify it matches the models before relying on it:
#
#     python manage.py makemigrations search_engine --check --dry-run   # expect "No changes detected"
#     python manage.py sqlmigrate search_engine 0011                    # inspect the emitted DDL
#
# WHY the custom DDL (SeparateDatabaseAndState + RunSQL):
#   A plain AlterField(IntegerField -> UUIDField) makes Django's Postgres backend
#   emit `ALTER COLUMN ... TYPE uuid USING <col>::uuid`. PostgreSQL has NO cast
#   from integer to uuid, and the USING expression is type-checked at plan time —
#   so it fails even on an empty / all-NULL table (the data-clear below removes
#   the rows but NOT the invalid cast expression). We therefore drive the schema
#   change with explicit `ALTER ... TYPE uuid USING NULL::uuid` (valid because the
#   columns hold only NULLs after the clear) inside `database_operations`, while
#   `state_operations` carries the AlterField so Django's migration state matches
#   the model and `makemigrations --check` stays green. SQLite (test DB) ignores
#   the type in ALTER, so this is also safe there.
#
# WHY the data-clear: the chat identity moved from the legacy integer
# (chat_type, chat_id, thread_id) tuple to v3 UUIDs (Channel.id / Message.id).
# Existing integer values reference the now-dropped legacy chat tables and are
# meaningless under the new scheme. ThreadSummary is a regenerable cache (flushed
# entirely); AgentSession thread-scoped rows are demoted to plain sessions (their
# thread binding referenced dead legacy ids). Note-scoped AgentSession rows
# (note_type/note_id) are untouched.

from django.db import migrations, models


def _clear_stale_chat_ids(apps, schema_editor):
    ThreadSummary = apps.get_model("search_engine", "ThreadSummary")
    AgentSession = apps.get_model("search_engine", "AgentSession")

    # ThreadSummary.chat_id / thread_id are NOT NULL ints — empty the cache so
    # the columns hold no integer values when the type changes.
    ThreadSummary.objects.all().delete()

    # AgentSession thread-scoped rows: null the chat binding so the nullable
    # columns hold only NULLs (which the ALTER ... USING NULL::uuid accepts).
    AgentSession.objects.filter(chat_id__isnull=False).update(
        chat_type=None, chat_id=None, thread_id=None
    )


# Postgres DDL: type-change driven by an explicit, valid USING expression.
# (SQLite silently ignores column-type changes, so this is a no-op there.)
_FORWARD_SQL = """
ALTER TABLE search_engine_agentsession  ALTER COLUMN chat_id   TYPE uuid USING NULL::uuid;
ALTER TABLE search_engine_agentsession  ALTER COLUMN thread_id TYPE uuid USING NULL::uuid;
ALTER TABLE search_engine_threadsummary ALTER COLUMN chat_id   TYPE uuid USING NULL::uuid;
ALTER TABLE search_engine_threadsummary ALTER COLUMN thread_id TYPE uuid USING NULL::uuid;
"""

_REVERSE_SQL = """
ALTER TABLE search_engine_agentsession  ALTER COLUMN chat_id   TYPE integer USING NULL::integer;
ALTER TABLE search_engine_agentsession  ALTER COLUMN thread_id TYPE integer USING NULL::integer;
ALTER TABLE search_engine_threadsummary ALTER COLUMN chat_id   TYPE integer USING NULL::integer;
ALTER TABLE search_engine_threadsummary ALTER COLUMN thread_id TYPE integer USING NULL::integer;
"""


class Migration(migrations.Migration):

    dependencies = [
        ("search_engine", "0010_agentrunjudgement"),
    ]

    operations = [
        # 1) Empty the stale integer values first (on the still-integer columns).
        migrations.RunPython(_clear_stale_chat_ids, migrations.RunPython.noop),
        # 2) Change the column types. DB side uses explicit valid DDL; state side
        #    records the field type so the model and migration state agree.
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(sql=_FORWARD_SQL, reverse_sql=_REVERSE_SQL),
            ],
            state_operations=[
                migrations.AlterField(
                    model_name="agentsession",
                    name="chat_id",
                    field=models.UUIDField(blank=True, null=True),
                ),
                migrations.AlterField(
                    model_name="agentsession",
                    name="thread_id",
                    field=models.UUIDField(blank=True, null=True),
                ),
                migrations.AlterField(
                    model_name="threadsummary",
                    name="chat_id",
                    field=models.UUIDField(),
                ),
                migrations.AlterField(
                    model_name="threadsummary",
                    name="thread_id",
                    field=models.UUIDField(),
                ),
            ],
        ),
    ]
