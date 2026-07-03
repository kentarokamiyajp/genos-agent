# Hand-authored to match models.AgentRunJudgement (F2 online judge
# sampling — SPOTLIGHT_QUALITY_ARCHITECTURE.md §F2). Django was not
# installed in the authoring environment, so this could not be generated
# or verified there. Verify it matches the model before relying on it:
#
#     python manage.py makemigrations search_engine --check --dry-run
#
# Expect "No changes detected". If it reports a change, delete this file
# and run `python manage.py makemigrations search_engine` to regenerate.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("search_engine", "0009_notesummary_agentsession_note_id_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="AgentRunJudgement",
            fields=[
                ("id", models.BigAutoField(primary_key=True, serialize=False)),
                ("team_id", models.CharField(db_index=True, max_length=64)),
                ("faithfulness", models.FloatField(default=0.0)),
                ("citation_precision", models.FloatField(default=0.0)),
                ("completeness", models.FloatField(default=0.0)),
                ("notes", models.TextField(blank=True, default="")),
                ("judge_model", models.CharField(blank=True, default="", max_length=64)),
                ("error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="judgements",
                        to="search_engine.agentrun",
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="agentrunjudgement",
            index=models.Index(
                fields=["team_id", "-created_at"], name="se_judge_team_created_idx"
            ),
        ),
    ]
