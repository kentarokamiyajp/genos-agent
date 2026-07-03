import uuid

from django.db import models
from django.utils import timezone


class RagChunk(models.Model):
    """Per-chunk tracking record.

    One row per OpenSearch document. Lets the indexer detect:
      - changed chunks (text_hash differs → re-embed + upsert)
      - stale chunks (chunk_id exists here but not in current
        regeneration of its parent entity → delete from OpenSearch)
      - re-embed needs after model upgrades (embedding_model differs)
    """

    chunk_id = models.CharField(primary_key=True, max_length=255)
    entity_type = models.CharField(max_length=32, db_index=True)
    entity_id = models.CharField(max_length=128, db_index=True)
    chunk_type = models.CharField(max_length=64)
    team_id = models.UUIDField(db_index=True)

    text_hash = models.CharField(max_length=64)
    source_version = models.BigIntegerField(blank=True, null=True)
    embedding_model = models.CharField(max_length=64)
    index_schema_version = models.CharField(max_length=16, default="v1")

    indexed_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["entity_type", "entity_id"]),
        ]


class AgentSession(models.Model):
    """Groups a sequence of /ask/ calls into one conversation.

    Phase 8 — conversation memory. When the frontend sends
    `session_id` with /ask/, the controller prepends the last
    SESSION_MAX_PRIOR_TURNS (query, final_answer) pairs into the
    model's context window before the current query. This allows
    follow-up references like "show me more about that task".

    TTL is enforced at load time via `last_active_at`. Sessions
    older than SESSION_TTL_MINUTES are silently retired and a new
    one is created. `last_active_at` is updated manually each time
    the session is successfully loaded.
    """

    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    team_id = models.CharField(max_length=64, db_index=True)
    user_id = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_active_at = models.DateTimeField(default=timezone.now)
    # ----- Thread Q&A scope (nullable; only set for sessions bound to
    # a specific chat thread via the "Ask about this thread" feature) -----
    # When all three are populated, the lookup in `_get_or_create_session`
    # treats this as a long-lived per-thread session, ignoring TTL — a
    # user might come back days later and reasonably expect their prior
    # Q&A to still be there. For regular Spotlight sessions, these stay
    # null and the existing TTL window applies.
    chat_type = models.IntegerField(blank=True, null=True)
    # v3 unified chat identity: `chat_id` is the `Channel.id` UUID and
    # `thread_id` is the thread-root `Message.id` UUID (was the legacy
    # integer chat_id / per-channel seq).
    chat_id = models.UUIDField(blank=True, null=True)
    thread_id = models.UUIDField(blank=True, null=True)
    # ----- Note Q&A scope (nullable; mirrors the thread scope above) -----
    # Set when the session is bound to a specific note via the
    # "Ask about this note" feature. Mutually exclusive with the chat
    # scope at the request layer — `_get_or_create_session` rejects
    # both. Same TTL bypass logic as the thread scope (line 132): a
    # user reopening a note Q&A days later should still see their prior
    # turns.
    note_type = models.IntegerField(blank=True, null=True)
    note_id = models.IntegerField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["team_id", "user_id", "-last_active_at"]),
            # Per-thread lookup: "do I have an existing thread session
            # for this user on this thread?" hits this index directly.
            models.Index(fields=["user_id", "chat_type", "chat_id", "thread_id"]),
            # Per-note lookup: same shape as the thread one.
            models.Index(fields=["user_id", "note_type", "note_id"]),
        ]


class AgentRun(models.Model):
    """One row per `/api/v2/agent/ask/` invocation.

    Status values:
        running             — loop is still in flight
        done                — clean exit, model produced a final answer
        error               — fatal mid-stream (Gemini failure, etc.)
        step_cap            — hit MAX_STEPS without a final answer
        awaiting_approval   — Phase 7: paused on a requires_approval
                              tool; resume via POST /api/v2/agent/decide/
        rejected            — Phase 7: user rejected the pending tool
                              call; loop resumed and produced a final
                              answer (terminal state, not a separate
                              flavor of `done`)

    `pending_approval_token` is a one-shot UUID emitted with the
    `tool_call_pending_approval` event and required (along with run_id)
    on the decide endpoint. The server clears it the moment the run
    leaves `awaiting_approval`, so a stale token can't be replayed.
    """

    run_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    team_id = models.CharField(max_length=64, db_index=True)
    user_id = models.CharField(max_length=64, db_index=True)
    query = models.TextField()
    status = models.CharField(max_length=20, default="running")
    final_answer_text = models.TextField(blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    # Phase 7 — write-tool approval flow.
    pending_approval_token = models.UUIDField(blank=True, null=True)
    # Phase 8 — conversation memory.
    session = models.ForeignKey(
        AgentSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="runs",
    )
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["team_id", "user_id", "-started_at"]),
        ]


class ThreadSummary(models.Model):
    """Cached LLM-generated summary of a chat thread, shared by all members.

    One row per (chat_type, chat_id, thread_id). The "Ask about this thread"
    feature checks the stored fingerprint against the live message fingerprint
    on every request: if they match, the cached summary is returned without
    re-invoking the LLM; if they differ (new message, edit, or delete), the
    summary is regenerated and the row updated in place.

    The summary is also indexed in OpenSearch via the thread_summary chunker
    so that the workspace-wide agent (Spotlight) can retrieve it.

    Fingerprint is `f"{max_thread_message_id}:{count_non_deleted}:{max_ts_updated_at}"`.
    Single-field timestamp keys miss edits and deletes; the three together catch
    inserts (bumps id+count), edits (bumps last_edit_ts), and deletes (drops count).
    """

    id = models.BigAutoField(primary_key=True)
    team_id = models.CharField(max_length=64, db_index=True)
    chat_type = models.IntegerField()  # 1=DM 2=GM 3=PM 4=MDM
    # v3 unified chat identity: `chat_id` is the `Channel.id` UUID and
    # `thread_id` is the thread-root `Message.id` UUID. `last_message_id`
    # stays an integer — it's the max per-channel `Message.seq` (still a
    # monotonic int) used in the fingerprint.
    chat_id = models.UUIDField()
    thread_id = models.UUIDField()
    summary_text = models.TextField()
    last_message_id = models.IntegerField(default=0)
    message_count = models.IntegerField(default=0)
    last_edit_ts = models.DateTimeField(blank=True, null=True)
    model_used = models.CharField(max_length=64, blank=True, default="")
    generated_by_user_id = models.CharField(max_length=64, blank=True, default="")
    ts_created_at = models.DateTimeField(auto_now_add=True)
    ts_updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["chat_type", "chat_id", "thread_id"],
                name="uq_thread_summary_scope",
            ),
        ]
        indexes = [
            models.Index(fields=["team_id", "ts_updated_at"]),
        ]


class NoteSummary(models.Model):
    """Cached LLM-generated summary of a note, shared by all users with ACL.

    One row per (note_type, note_id). The "Ask about this note" feature
    checks the stored fingerprint against the live note row on every
    request: if they match, the cached summary is returned without
    re-invoking the LLM; if they differ (body edit, title edit), the
    summary is regenerated and the row updated in place.

    The summary is also indexed in OpenSearch via the note_summary
    chunker so that the workspace-wide agent (Spotlight) can retrieve
    it alongside the per-section note_chunker output.

    Fingerprint is `f"{ts_updated.isoformat()}:{body_length}:{title}"`.
    A pure `ts_updated` key would catch any save, but the body-length
    and title are cheap signals that defend against clock skew (clock
    going backwards on a master swap) — they both come from the live
    note row at fingerprint-compute time, so a stale cached row will
    only match when the title AND the body length AND the timestamp
    all match.
    """

    id = models.BigAutoField(primary_key=True)
    team_id = models.CharField(max_length=64, db_index=True)
    note_type = models.IntegerField()  # 1=Personal, 2=Task, 3=Chat
    note_id = models.IntegerField()
    summary_text = models.TextField()
    # Fingerprint inputs (see compute_fingerprint in note_summary.py).
    last_edit_ts = models.DateTimeField(blank=True, null=True)
    body_length = models.IntegerField(default=0)
    title_at_gen = models.CharField(max_length=512, blank=True, default="")
    model_used = models.CharField(max_length=64, blank=True, default="")
    generated_by_user_id = models.CharField(max_length=64, blank=True, default="")
    ts_created_at = models.DateTimeField(auto_now_add=True)
    ts_updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["note_type", "note_id"],
                name="uq_note_summary_scope",
            ),
        ]
        indexes = [
            models.Index(fields=["team_id", "ts_updated_at"]),
        ]


class AgentStep(models.Model):
    """One row per step within an `AgentRun`.

    A step is either a tool-call (tool_name + arguments_json + result_json
    populated) or a text-only model turn (answer_text populated). The
    `result_json` field holds the full tool output and is intentionally
    server-side only — only `summary` ever reaches the client.
    """

    step_id = models.AutoField(primary_key=True)
    run = models.ForeignKey(AgentRun, on_delete=models.CASCADE, related_name="steps")
    step_index = models.IntegerField()
    tool_name = models.CharField(max_length=64, blank=True, default="")
    arguments_json = models.JSONField(blank=True, null=True)
    summary = models.TextField(blank=True, default="")
    result_json = models.JSONField(blank=True, null=True)
    answer_text = models.TextField(blank=True, default="")
    error = models.TextField(blank=True, default="")
    # Opaque Gemini 3+ "thought signature" bytes captured alongside a
    # function_call part. Must be echoed back when the assistant turn is
    # replayed (e.g. after a write-tool approval resume) or Gemini 3
    # rejects with `400 INVALID_ARGUMENT: Function call is missing a
    # thought_signature in functionCall parts.` See
    # `FunctionCall.thought_signature` in llm/types.py.
    thought_signature = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["run", "step_index"]),
        ]


class AgentRunJudgement(models.Model):
    """Online quality sample (F2 — SPOTLIGHT_QUALITY_ARCHITECTURE.md §F2).

    One LLM-judge scoring of a completed `AgentRun`, produced
    ASYNCHRONOUSLY by the `agent_judge_sample` management command — never
    on the user request path. Lets us trend *production* faithfulness /
    citation_precision / completeness and alert on drift, which the fixed
    offline eval suite (37 cases) can't see.

    One run is judged at most once: the sampler excludes runs that
    already have a judgement, so re-running the cron is idempotent.
    `error` is non-empty when the judge call itself failed; the three
    scores are 0.0 in that case (mirrors `judge._error_scores`) so a
    failed judgement is recorded rather than silently dropped, and the
    `--report` aggregator filters `error=""` so failures don't drag the
    mean.
    """

    id = models.BigAutoField(primary_key=True)
    run = models.ForeignKey(AgentRun, on_delete=models.CASCADE, related_name="judgements")
    # Denormalized from the run so per-team rollups don't need a join.
    team_id = models.CharField(max_length=64, db_index=True)
    faithfulness = models.FloatField(default=0.0)
    citation_precision = models.FloatField(default=0.0)
    completeness = models.FloatField(default=0.0)
    notes = models.TextField(blank=True, default="")
    # Which model produced this score — recorded for reproducibility when
    # the active provider/model changes between sampling passes.
    judge_model = models.CharField(max_length=64, blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            # Explicit name (not Django's auto hash) so the hand-authored
            # migration is deterministically verifiable via
            # `makemigrations --check`.
            models.Index(fields=["team_id", "-created_at"], name="se_judge_team_created_idx"),
        ]


class AgentRunFeedback(models.Model):
    """Human 👍/👎 on an agent answer (F1 — SPOTLIGHT_QUALITY_ARCHITECTURE.md
    §Q0). The doc flags this signal as "genuinely absent — no model, no
    endpoint, no UI"; it is the reward signal D5 (inline-citation preference),
    D4 (RLHF/DPO), and F3 (bandit config selection) all gate on.

    One row per (run, user): a given user's verdict on a given answer.
    Recorded on POST /api/v2/agent/runs/<run_id>/feedback/ — NEVER on the
    answer path. `rating` is +1 (👍) / -1 (👎); 0 is the explicit "cleared"
    state (the UI toggles a vote back off). `comment` is optional free text
    for a future "tell us why" affordance.
    """

    RATING_UP = 1
    RATING_DOWN = -1
    RATING_CLEARED = 0

    id = models.BigAutoField(primary_key=True)
    run = models.ForeignKey(AgentRun, on_delete=models.CASCADE, related_name="feedback")
    # Denormalized from the run so per-team rollups don't need a join
    # (mirrors AgentRunJudgement.team_id).
    team_id = models.CharField(max_length=64, db_index=True)
    user_id = models.CharField(max_length=64, db_index=True)
    rating = models.SmallIntegerField()  # -1 | 0 | +1
    comment = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "user_id"], name="uq_agent_feedback_run_user"
            ),
        ]
        indexes = [
            models.Index(fields=["team_id", "-created_at"], name="se_feedback_team_created_idx"),
        ]
