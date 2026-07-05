# Hand-authored (verified with `makemigrations --check`), like 0010/0012.
#
# D5 (§4.6) — adds the nullable prose-faithfulness axis to the online
# judge sample. NULL means "this run's answer had no link-form
# citations to score", which is semantically different from 0.0
# ("the link text lied about the source") — that's why the column is
# nullable with no default backfill.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("search_engine", "0012_agentrunfeedback"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentrunjudgement",
            name="prose_faithfulness",
            field=models.FloatField(blank=True, default=None, null=True),
        ),
    ]
