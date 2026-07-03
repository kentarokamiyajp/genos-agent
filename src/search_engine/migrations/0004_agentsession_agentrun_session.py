import uuid

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("search_engine", "0003_agentrun_pending_approval_token"),
    ]

    operations = [
        migrations.CreateModel(
            name="AgentSession",
            fields=[
                (
                    "session_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("team_id", models.CharField(db_index=True, max_length=64)),
                ("user_id", models.CharField(db_index=True, max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "last_active_at",
                    models.DateTimeField(default=django.utils.timezone.now),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["team_id", "user_id", "-last_active_at"],
                        name="se_sess_tu_last_idx",
                    )
                ],
            },
        ),
        migrations.AddField(
            model_name="agentrun",
            name="session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="runs",
                to="search_engine.agentsession",
            ),
        ),
    ]
