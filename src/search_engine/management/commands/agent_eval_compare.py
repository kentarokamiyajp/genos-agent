"""`python manage.py agent_eval_compare` — A/B two configs on one suite.

Runs the eval suite twice — once with the current settings, once with
a JSON-encoded override applied to `SEARCH_ENGINE` — and prints a
per-case diff plus aggregate deltas. Use to validate a tuning change
("does enabling the reranker improve pass rate?") without committing
to a permanent settings flip.

Both runs share the same fixture, so the only variable is the
override. Order is fixed (A first, then B), so anything time-sensitive
in the runner doesn't drift between the two.

Examples
--------

Compare baseline vs reranker on, retrieval suite:

    python manage.py agent_eval_compare --retrieval \
        --b-overrides '{"RAG_USE_RERANKER": true, "RAG_RERANK_OUTPUT_K": 5}'

Compare baseline vs query-rewrite on, behavior suite with judge:

    python manage.py agent_eval_compare --judge \
        --b-overrides '{"RAG_USE_QUERY_REWRITE": true}'

Notes
-----
* Overrides are applied to `django.conf.settings.SEARCH_ENGINE` in
  place. The command restores the original values on exit (even on
  error) so a partial run can't poison subsequent commands in the
  same shell session.
* This is a development tool; it does not aggregate into the JSONL
  run log written by `agent_eval --judge`. If you want a permanent
  record of an A/B, run the judge separately under each config.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand

from origin.search_engine.agent.evals.judge import judge_answer
from origin.search_engine.agent.evals.runner import (
    BEHAVIOR_CASES_PATH,
    RETRIEVAL_CASES_PATH,
    CaseResult,
    load_cases,
    run_behavior_case,
    run_retrieval_case,
)


class Command(BaseCommand):
    help = "Run an eval suite twice under two configs and diff the results."

    def add_arguments(self, parser):
        parser.add_argument(
            "--retrieval",
            action="store_true",
            help="Compare on the retrieval suite (fast, no LLM calls).",
        )
        parser.add_argument(
            "--judge",
            action="store_true",
            help=(
                "Behavior-suite only. Score each run's answers with "
                "the LLM judge so the diff shows quality deltas."
            ),
        )
        parser.add_argument(
            "--b-overrides",
            dest="b_overrides",
            required=True,
            help=(
                "JSON dict of SEARCH_ENGINE keys → values applied to "
                "the B run. Example: '{\"RAG_USE_RERANKER\": true}'."
            ),
        )
        parser.add_argument(
            "--case",
            dest="case_id",
            default=None,
            help="Limit to a single case (matches `id:` in the YAML).",
        )

    def handle(self, *args, **options):
        retrieval_only: bool = options.get("retrieval") or False
        run_judge: bool = options.get("judge") or False
        case_id_filter: str | None = options.get("case_id")

        if run_judge and retrieval_only:
            self.stderr.write(
                self.style.ERROR(
                    "--judge does not apply to the retrieval suite (no answer to score)."
                )
            )
            sys.exit(2)

        try:
            overrides: dict[str, Any] = json.loads(options["b_overrides"])
            if not isinstance(overrides, dict):
                raise ValueError("must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self.stderr.write(self.style.ERROR(f"Invalid --b-overrides: {exc}"))
            sys.exit(2)

        # Pick suite + runner.
        if retrieval_only:
            path = RETRIEVAL_CASES_PATH
            runner = run_retrieval_case
            label = "retrieval"
        else:
            path = BEHAVIOR_CASES_PATH
            runner = run_behavior_case
            label = "behavior"

        cases = load_cases(path)
        if case_id_filter:
            cases = [c for c in cases if c.get("id") == case_id_filter]
            if not cases:
                self.stderr.write(self.style.ERROR(f"No case with id={case_id_filter!r}"))
                sys.exit(2)

        self.stdout.write(
            f"=== {label} suite: A (baseline) vs B (with overrides) ===\n"
            f"B overrides: {overrides}\n"
            f"Running {len(cases)} cases × 2 = {2 * len(cases)} runs.\n"
        )

        # --- A run: untouched settings ---
        self.stdout.write(self.style.NOTICE("\n--- A (baseline) ---"))
        results_a = self._run_suite(cases, runner, label, run_judge)

        # --- B run: settings with overrides applied, restored after ---
        self.stdout.write(self.style.NOTICE("\n--- B (with overrides) ---"))
        with _override_search_engine(overrides):
            results_b = self._run_suite(cases, runner, label, run_judge)

        self._print_diff(results_a, results_b, run_judge=run_judge)

    def _run_suite(
        self,
        cases: list[dict[str, Any]],
        runner,
        label: str,
        run_judge: bool,
    ) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in cases:
            # `runner` is destructive — `_resolve_fixture` mutates the
            # case dict. Copy first so the B run starts from the same
            # YAML shape as A did.
            r = runner(dict(case))
            if run_judge and label == "behavior" and r.answer:
                r.judge_scores = judge_answer(
                    query=r.query,
                    sources=r.sources,
                    answer=r.answer,
                    tool_results=r.tool_results,
                )
            results.append(r)
            self.stdout.write(
                f"  {'PASS' if r.passed else 'FAIL'}  {r.case_id:<48} ({r.duration_ms} ms)"
            )
        return results

    def _print_diff(
        self,
        a: list[CaseResult],
        b: list[CaseResult],
        *,
        run_judge: bool,
    ) -> None:
        by_id_a = {r.case_id: r for r in a}
        by_id_b = {r.case_id: r for r in b}
        all_ids = sorted(set(by_id_a) | set(by_id_b))

        moved: list[tuple[str, str]] = []  # (case_id, "A→B" change description)
        for cid in all_ids:
            ra = by_id_a.get(cid)
            rb = by_id_b.get(cid)
            if ra and rb and ra.passed != rb.passed:
                arrow = "PASS→FAIL" if ra.passed else "FAIL→PASS"
                moved.append((cid, arrow))

        self.stdout.write("\n=== diff ===")
        passed_a = sum(1 for r in a if r.passed)
        passed_b = sum(1 for r in b if r.passed)
        delta_str = (
            self.style.SUCCESS(f"+{passed_b - passed_a}")
            if passed_b > passed_a
            else (self.style.ERROR(f"{passed_b - passed_a}") if passed_b < passed_a else "0")
        )
        self.stdout.write(
            f"  Pass count: A={passed_a}/{len(a)}  B={passed_b}/{len(b)}  delta {delta_str}"
        )

        if moved:
            self.stdout.write("  Cases that changed verdict:")
            for cid, arrow in moved:
                style = self.style.SUCCESS if arrow == "FAIL→PASS" else self.style.ERROR
                self.stdout.write(f"    {style(arrow):>10}  {cid}")
        else:
            self.stdout.write("  No case changed verdict.")

        # Latency aggregate.
        dur_a = sum(r.duration_ms for r in a) / max(1, len(a))
        dur_b = sum(r.duration_ms for r in b) / max(1, len(b))
        delta_pct = (dur_b - dur_a) / max(1.0, dur_a) * 100
        self.stdout.write(
            f"  Mean latency: A={dur_a:.0f} ms  B={dur_b:.0f} ms  ({delta_pct:+.0f}%)"
        )

        # Continuous-metric deltas (mrr / recall_at_n for retrieval;
        # tool_recall / tool_excl_ok for tool selection). The payoff of
        # the metrics: surfaces gains that DON'T cross the binary
        # pass/fail line — e.g. a change that lifts gold from rank 4 to
        # rank 2 shows as +mrr while "Pass count" stays flat. Each metric
        # is averaged over the cases that declared it, so the (n=) differs
        # per row.
        metric_a = [r for r in a if r.metrics]
        metric_b = [r for r in b if r.metrics]
        if metric_a and metric_b:
            keys = sorted({k for r in metric_a + metric_b for k in r.metrics})

            def _vals(rs: list[CaseResult], key: str) -> list[float]:
                return [r.metrics[key] for r in rs if key in r.metrics]

            for k in keys:
                va = _vals(metric_a, k)
                vb = _vals(metric_b, k)
                av = sum(va) / len(va) if va else 0.0
                bv = sum(vb) / len(vb) if vb else 0.0
                delta = bv - av
                arrow = (
                    self.style.SUCCESS(f"{delta:+.3f}")
                    if delta > 0.005
                    else (self.style.ERROR(f"{delta:+.3f}") if delta < -0.005 else f"{delta:+.3f}")
                )
                self.stdout.write(
                    f"  metric.{k:<16} A={av:.3f}  B={bv:.3f}  delta {arrow}  (n={len(va)})"
                )

        if run_judge:
            judged_a = [r for r in a if r.judge_scores]
            judged_b = [r for r in b if r.judge_scores]
            if judged_a and judged_b:

                def _avg(rs: list[CaseResult], key: str) -> float:
                    return sum(r.judge_scores.get(key, 0.0) for r in rs) / len(rs)

                for k in ("faithfulness", "citation_precision", "completeness"):
                    av = _avg(judged_a, k)
                    bv = _avg(judged_b, k)
                    delta = bv - av
                    arrow = (
                        self.style.SUCCESS(f"{delta:+.2f}")
                        if delta > 0.005
                        else (
                            self.style.ERROR(f"{delta:+.2f}")
                            if delta < -0.005
                            else f"{delta:+.2f}"
                        )
                    )
                    self.stdout.write(f"  judge.{k:<20} A={av:.2f}  B={bv:.2f}  delta {arrow}")


class _override_search_engine:
    """Context manager: apply overrides to `settings.SEARCH_ENGINE`, restore on exit."""

    def __init__(self, overrides: dict[str, Any]):
        self.overrides = overrides
        self._sentinel = object()
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        cfg = settings.SEARCH_ENGINE
        for k, v in self.overrides.items():
            self._saved[k] = cfg.get(k, self._sentinel)
            cfg[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        cfg = settings.SEARCH_ENGINE
        for k, prev in self._saved.items():
            if prev is self._sentinel:
                cfg.pop(k, None)
            else:
                cfg[k] = prev
        return False
