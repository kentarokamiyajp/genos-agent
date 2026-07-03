# Generated for Gemini 3 thought_signature persistence — see
# AgentStep.thought_signature and llm/types.py FunctionCall for context.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("search_engine", "0005_rename_se_sess_tu_last_idx_search_engi_team_id_8962f4_idx"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentstep",
            name="thought_signature",
            field=models.BinaryField(blank=True, null=True),
        ),
    ]
