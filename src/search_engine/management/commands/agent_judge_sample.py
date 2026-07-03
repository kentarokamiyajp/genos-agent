"""`python manage.py agent_judge_sample` — F2 online judge sampling.

SPOTLIGHT_QUALITY_ARCHITECTURE.md §F2. Samples a fraction of completed
production `AgentRun`s, scores each with the LLM judge (the same one the
offline suite uses), and persists the scores to `AgentRunJudgement` so
production faithfulness / citation_precision / completeness can be
*trended* — something the fixed offline eval suite can't see.

Runs entirely OFF the user request path: there is no async/Celery infra
in this project, so this is a cron-driven command. Point a scheduler at
it (e.g. hourly):

    0 * * * *  cd /app/backend_django && python manage.py agent_judge_sample

Two modes:

  * **Sample** (default) — judge a deterministic sample of recent,
    not-yet-judged runs and store the scores.

        python manage.py agent_judge_sample                 # rate from settings
        python manage.py agent_judge_sample --rate 0.2      # override rate
        python manage.py agent_judge_sample --dry-run       # show selection only

  * **Report** (`--report`) — print the trend of stored judgements,
    grouped by day, so the persisted data is readable by default:

        python manage.py agent_judge_sample --report
        python manage.py agent_judge_sample --report --report-days 30 --team <id>

Sampling is **deterministic by `run_id` hash**, so the effective sample
rate matches `RAG_JUDGE_SAMPLE_RATE` no matter how often the cron fires
(a per-pass `random()` would re-roll un-judged runs every pass and drive
the effective rate toward 1.0). Each pass is idempotent: runs that
already have a judgement are excluded.

Scope (sample validity — stated so the metric is honestly labelled):
  * Only `status="done"` runs are judged — the clean-completion signal.
    `rejected` / `step_cap` / `error` runs are excluded.
  * Runs with NO grounding (empty reconstructed sources AND no tool
    results — e.g. answered purely from prior context, or an honest
    "couldn't find that") are skipped: the judge has nothing to ground
    against and would inject noise. Abstention / prior-context quality is
    a separate measurement surface (the abstention axis).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from origin.management.cron_command import CronCommand
from origin.search_engine.agent.controller import reconstruct_sources_for_run
from origin.search_engine.agent.evals.judge import judge_answer
from origin.search_engine.models import AgentRun, AgentRunJudgement

log = logging.getLogger(__name__)

# Fine-grained bucketing for the deterministic hash sampler — gives the
# nominal rate ~6 significant figures of resolution.
_HASH_BUCKETS = 1_000_000


class Command(CronCommand):
    help = "Sample completed AgentRuns and score them with the LLM judge (F2 online eval)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--rate",
            type=float,
            default=None,
            help=(
                "Sampling fraction 0.0–1.0. Overrides "
                "SEARCH_ENGINE['RAG_JUDGE_SAMPLE_RATE'] for this pass."
            ),
        )
        parser.add_argument(
            "--since-hours",
            type=int,
            default=24,
            help="Only consider runs started within this many hours (default 24).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Max runs to judge this pass — caps LLM cost (default 50).",
        )
        parser.add_argument(
            "--team",
            dest="team_id",
            default=None,
            help="Restrict to a single team_id.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show which runs would be judged without calling the judge or writing rows.",
        )
        parser.add_argument(
            "--report",
            action="store_true",
            help="Print the trend of stored judgements (grouped by day) instead of sampling.",
        )
        parser.add_argument(
            "--report-days",
            type=int,
            default=7,
            help="Lookback window for --report (default 7 days).",
        )
        parser.add_argument(
            "--alert",
            action="store_true",
            help=(
                "With --report: exit non-zero if drift is detected (the recent "
                "half of the window dropped more than RAG_JUDGE_DRIFT_THRESHOLD "
                "below the prior half on any axis). For cron/CI paging."
            ),
        )

    def handle(self, *args, **options):
        if options.get("report"):
            drift = self._report(
                days=options["report_days"],
                team_id=options.get("team_id"),
                alert=options.get("alert") or False,
            )
            if options.get("alert") and drift:
                sys.exit(1)
            return

        rate = options["rate"]
        if rate is None:
            rate = float(settings.SEARCH_ENGINE.get("RAG_JUDGE_SAMPLE_RATE", 0.0))
        rate = max(0.0, min(1.0, rate))
        if rate <= 0.0:
            self.stdout.write(
                self.style.WARNING(
                    "Judge sampling disabled (rate=0.0). Set RAG_JUDGE_SAMPLE_RATE>0 "
                    "or pass --rate to enable."
                )
            )
            return

        since_hours: int = options["since_hours"]
        limit: int = options["limit"]
        team_id: str | None = options.get("team_id")
        dry_run: bool = options.get("dry_run") or False

        cutoff = timezone.now() - timedelta(hours=since_hours)
        # `judgements__isnull=True` → runs with no judgement yet (idempotent
        # across passes). `done` + non-empty answer = a clean, gradable
        # completion.
        base = (
            AgentRun.objects.filter(status="done", started_at__gte=cutoff)
            .exclude(final_answer_text="")
            .filter(judgements__isnull=True)
            .order_by("-started_at")
        )
        if team_id:
            base = base.filter(team_id=team_id)

        # Sample on run_id first (ids only), THEN hydrate just the winners
        # with their steps — avoids prefetching steps for every candidate
        # when we only judge up to `limit`.
        threshold = int(rate * _HASH_BUCKETS)
        sampled_ids: list[Any] = []
        for rid in base.values_list("run_id", flat=True):
            if self._in_sample(rid, threshold):
                sampled_ids.append(rid)
                if len(sampled_ids) >= limit:
                    break

        sampled = list(
            AgentRun.objects.filter(run_id__in=sampled_ids)
            .prefetch_related("steps")
            .order_by("-started_at")
        )

        self.stdout.write(
            f"Selected {len(sampled)} run(s) to judge "
            f"(rate={rate:.3f}, window={since_hours}h, cap {limit})"
            + (f", team={team_id}" if team_id else "")
        )

        if dry_run:
            for run in sampled:
                self.stdout.write(f"  would judge {run.run_id}  ({run.query[:70]!r})")
            self.stdout.write(
                self.style.NOTICE(f"\n--dry-run: {len(sampled)} run(s), nothing written.")
            )
            return

        judge_model = self._active_judge_model()
        judged = 0
        skipped_ungrounded = 0
        errors = 0
        score_sums = {"faithfulness": 0.0, "citation_precision": 0.0, "completeness": 0.0}

        for run in sampled:
            sources = reconstruct_sources_for_run(run)
            tool_results = [
                {
                    "tool_name": s.tool_name,
                    "arguments": s.arguments_json,
                    "result": s.result_json,
                }
                for s in run.steps.all()
                if s.tool_name and s.result_json is not None
            ]
            if not sources and not tool_results:
                skipped_ungrounded += 1
                continue

            scores = judge_answer(
                query=run.query,
                sources=sources,
                answer=run.final_answer_text,
                tool_results=tool_results,
            )
            err = scores.get("error") or ""
            AgentRunJudgement.objects.create(
                run=run,
                team_id=run.team_id,
                faithfulness=float(scores.get("faithfulness", 0.0)),
                citation_precision=float(scores.get("citation_precision", 0.0)),
                completeness=float(scores.get("completeness", 0.0)),
                notes=scores.get("notes", "") or "",
                judge_model=judge_model,
                error=err,
            )
            judged += 1
            if err:
                errors += 1
            else:
                for k in score_sums:
                    score_sums[k] += float(scores.get(k, 0.0))

        ok = judged - errors
        self.stdout.write(
            self.style.SUCCESS(
                f"\nJudged {judged} run(s) "
                f"({errors} judge error(s), {skipped_ungrounded} skipped — no grounding)."
            )
        )
        if errors:
            # Surface at ERROR so the CronCommand tripwire marks the cron run
            # failed instead of green when judge calls erred.
            log.error("agent_judge_sample: %d judge error(s) out of %d judged", errors, judged)
        if ok:
            self.stdout.write(
                self.style.NOTICE(
                    f"Means over {ok} scored: "
                    f"faith={score_sums['faithfulness'] / ok:.2f}  "
                    f"cite={score_sums['citation_precision'] / ok:.2f}  "
                    f"compl={score_sums['completeness'] / ok:.2f}"
                )
            )

    @staticmethod
    def _in_sample(run_id: Any, threshold: int) -> bool:
        """Deterministic per-run inclusion. `hashlib.sha1` (not builtin
        `hash`, which is PYTHONHASHSEED-salted per process) so the same
        run gets the same verdict across cron invocations."""
        digest = hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()
        return (int(digest, 16) % _HASH_BUCKETS) < threshold

    @staticmethod
    def _active_judge_model() -> str:
        """Best-effort label of the model the judge runs on. The judge
        calls `get_model_client()` with no override and the command has no
        per-user LlmChoice, so it resolves to the env-default model."""
        se = settings.SEARCH_ENGINE
        provider = (se.get("LLM_PROVIDER") or "gemini").lower()
        model = se.get("GEMINI_MODEL") if provider == "gemini" else se.get("CLAUDE_MODEL")
        return f"{provider}:{model or '?'}"

    def _report(self, *, days: int, team_id: str | None, alert: bool = False) -> bool:
        """Print the judgement trend; return True if drift was detected
        (recent-half mean dropped > RAG_JUDGE_DRIFT_THRESHOLD below prior-half
        on any axis). The bool drives the `--alert` exit code."""
        cutoff = timezone.now() - timedelta(days=days)
        qs = AgentRunJudgement.objects.filter(created_at__gte=cutoff)
        if team_id:
            qs = qs.filter(team_id=team_id)

        total = qs.count()
        if not total:
            self.stdout.write(
                self.style.WARNING(
                    f"No judgements in the last {days} day(s)"
                    + (f" for team {team_id}" if team_id else "")
                    + ". Run sampling first (RAG_JUDGE_SAMPLE_RATE>0)."
                )
            )
            return False

        # Means over successful judgements only (error="") so failed judge
        # calls (scored 0.0) don't drag the trend down.
        ok = qs.filter(error="")
        by_day = (
            ok.annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(
                n=Count("id"),
                faith=Avg("faithfulness"),
                cite=Avg("citation_precision"),
                compl=Avg("completeness"),
            )
            .order_by("day")
        )

        self.stdout.write(
            f"=== Judge trend — last {days} day(s)"
            + (f", team {team_id}" if team_id else "")
            + f" ({total} judgement(s), {qs.filter(~Q(error='')).count()} errored) ==="
        )
        self.stdout.write(f"  {'day':<12} {'n':>4}  {'faith':>6} {'cite':>6} {'compl':>6}")
        for row in by_day:
            self.stdout.write(
                f"  {str(row['day']):<12} {row['n']:>4}  "
                f"{row['faith']:>6.2f} {row['cite']:>6.2f} {row['compl']:>6.2f}"
            )

        agg = ok.aggregate(
            faith=Avg("faithfulness"),
            cite=Avg("citation_precision"),
            compl=Avg("completeness"),
            n=Count("id"),
        )
        if agg["n"]:
            self.stdout.write(
                self.style.NOTICE(
                    f"\n  overall ({agg['n']} scored): "
                    f"faith={agg['faith']:.2f}  cite={agg['cite']:.2f}  compl={agg['compl']:.2f}"
                )
            )

        return self._check_drift(ok, days=days, team_id=team_id)

    # Axes the drift check watches, mapped to their model field.
    _DRIFT_AXES = (
        ("faith", "faithfulness"),
        ("cite", "citation_precision"),
        ("compl", "completeness"),
    )
    # Both windows need at least this many scored runs for the comparison to
    # be meaningful (a 2-run window swings on noise, not drift).
    _DRIFT_MIN_PER_WINDOW = 5

    def _check_drift(self, ok_qs, *, days: int, team_id: str | None) -> bool:
        """Compare the recent half of the lookback against the prior half and
        flag any axis that regressed beyond the threshold (F2 'alert on
        regression'). `ok_qs` is the error-free judgement queryset."""
        threshold = float(settings.SEARCH_ENGINE.get("RAG_JUDGE_DRIFT_THRESHOLD", 0.05))
        recent_days = max(1, days // 2)
        split = timezone.now() - timedelta(days=recent_days)

        recent = ok_qs.filter(created_at__gte=split).aggregate(
            n=Count("id"),
            faith=Avg("faithfulness"),
            cite=Avg("citation_precision"),
            compl=Avg("completeness"),
        )
        prior = ok_qs.filter(created_at__lt=split).aggregate(
            n=Count("id"),
            faith=Avg("faithfulness"),
            cite=Avg("citation_precision"),
            compl=Avg("completeness"),
        )

        self.stdout.write(
            f"\n=== Drift check — recent {recent_days}d (n={recent['n']}) "
            f"vs prior {days - recent_days}d (n={prior['n']}), "
            f"threshold {threshold:.2f} ==="
        )
        if recent["n"] < self._DRIFT_MIN_PER_WINDOW or prior["n"] < self._DRIFT_MIN_PER_WINDOW:
            self.stdout.write(
                self.style.WARNING(
                    f"  insufficient data (need >= {self._DRIFT_MIN_PER_WINDOW} scored "
                    "runs per window); skipping drift verdict."
                )
            )
            return False

        drifted: list[str] = []
        for short, _field in self._DRIFT_AXES:
            r = recent[short] or 0.0
            p = prior[short] or 0.0
            delta = r - p
            regressed = delta < -threshold
            mark = self.style.ERROR("DRIFT") if regressed else self.style.SUCCESS("ok   ")
            self.stdout.write(f"  {mark} {short:<6} prior={p:.2f} → recent={r:.2f}  Δ={delta:+.2f}")
            if regressed:
                drifted.append(f"{short} {p:.2f}→{r:.2f} (Δ{delta:+.2f})")

        if drifted:
            msg = (
                "judge drift detected"
                + (f" for team {team_id}" if team_id else "")
                + ": "
                + "; ".join(drifted)
            )
            log.warning("F2 %s", msg)
            self.stdout.write(self.style.ERROR(f"\n  → {msg}"))
            return True
        return False
