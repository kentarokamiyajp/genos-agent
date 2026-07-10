"""`python manage.py agent_eval` — run the agent eval harness.

Two suite types, sharing the same CLI shape:

  * **Behavior** (default) — full agent loop, real Gemini/Claude calls,
    asserts on the NDJSON event stream. Source: `agent/evals/cases.yaml`.

  * **Retrieval** (`--retrieval`) — direct `search(...)` calls, no LLM,
    asserts on the ranked entity list. Source:
    `agent/evals/retrieval_cases.yaml`.

Exit code is 0 if every case in the chosen suite(s) passed, 1
otherwise (CI-friendly).

Examples:

    python manage.py agent_eval                       # behavior suite
    python manage.py agent_eval --retrieval           # retrieval suite (fast, no LLM)
    python manage.py agent_eval --all                 # both suites
    python manage.py agent_eval --case <id>           # one case (auto-detects suite)
    python manage.py agent_eval --retrieval --fail-fast
    python manage.py agent_eval --judge               # behavior + LLM judge
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from django.core.management.base import BaseCommand

from origin.search_engine.agent.evals.judge import judge_answer
from origin.search_engine.agent.evals.runner import (
    BEHAVIOR_CASES_PATH,
    RETRIEVAL_CASES_PATH,
    SUMMARY_CASES_PATH,
    CaseResult,
    load_cases,
    run_behavior_case,
    run_retrieval_case,
    run_summary_case,
)

RUNS_DIR = Path(__file__).resolve().parents[3] / "agent" / "evals" / "runs"


class Command(BaseCommand):
    help = "Run the agent evaluation harness (behavior and/or retrieval suite)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--case",
            dest="case_id",
            default=None,
            help="Run only the case with this id (matches `id:` in cases.yaml).",
        )
        parser.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop on the first failing case.",
        )
        suite_group = parser.add_mutually_exclusive_group()
        suite_group.add_argument(
            "--retrieval",
            action="store_true",
            help="Run only the retrieval suite (fast, no LLM calls).",
        )
        suite_group.add_argument(
            "--summary",
            dest="summary_only",
            action="store_true",
            help=(
                "Run only the summary-fidelity suite (Q0.3): summarise each "
                "source with the production summary prompt, then score "
                "summary-vs-source fidelity / coverage / entity-preservation. "
                "Source: agent/evals/summary_cases.yaml."
            ),
        )
        suite_group.add_argument(
            "--all",
            dest="run_all",
            action="store_true",
            help="Run behavior, retrieval, and summary suites.",
        )
        parser.add_argument(
            "--judge",
            action="store_true",
            help=(
                "Score each behavior case's answer with an LLM judge "
                "(faithfulness / citation_precision / completeness). "
                "Adds ~1 LLM call per case; results persisted to "
                "agent/evals/runs/<timestamp>.jsonl."
            ),
        )
        parser.add_argument(
            "--persist-metrics",
            action="store_true",
            help=(
                "Write per-case continuous metrics (mrr / recall_at_n) to "
                "agent/evals/runs/<timestamp>-metrics.jsonl so retrieval "
                "quality can be trended across PRs. Opt-in so the fast "
                "retrieval suite doesn't spam files during iteration."
            ),
        )
        parser.add_argument(
            "--skip-env-check",
            action="store_true",
            help=(
                "Skip the index-mapping drift preflight. The eval refuses to "
                "run when the live search index's mapping is missing subfields "
                "the code queries (e.g. a local index created before the "
                "multilingual .icu/.ja or .en/.prefix subfields landed) — a "
                "stale index silently depresses every retrieval number and "
                "turns an assessment into noise (quality round 2 found a "
                "local index with ZERO subfields behind 4 'chronic' FAILs). "
                "Fix: opensearch_setup --recreate && opensearch_reindex && "
                "agent_eval_setup --reseed."
            ),
        )
        parser.add_argument(
            "--metric-gate",
            action="store_true",
            help=(
                "ENFORCE the tool-selection north-star (Q0.4): exit 1 when the "
                "aggregate tool_recall / tool_excl_ok is below "
                "RAG_TOOL_SELECTION_NORTH_STAR. Default is observe-only — the "
                "north-star line is always reported, but the run only fails on "
                "it when this flag is set (the fixture gold for the multi-tool "
                "cases is still being validated, so enforcing-by-default would "
                "regress a suite the team runs green)."
            ),
        )

    def handle(self, *args, **options):
        case_id_filter: str | None = options.get("case_id")
        fail_fast: bool = options.get("fail_fast") or False
        retrieval_only: bool = options.get("retrieval") or False
        summary_only: bool = options.get("summary_only") or False
        run_all: bool = options.get("run_all") or False
        run_judge: bool = options.get("judge") or False
        persist_metrics: bool = options.get("persist_metrics") or False
        metric_gate: bool = options.get("metric_gate") or False

        if not options.get("skip_env_check"):
            drift = self._index_mapping_drift()
            if drift:
                self.stderr.write(
                    self.style.ERROR(
                        "Search index mapping is STALE — eval numbers would be noise:\n  "
                        + "\n  ".join(drift)
                        + "\nFix:  python manage.py opensearch_setup --recreate && "
                        "python manage.py opensearch_reindex && "
                        "python manage.py agent_eval_setup --reseed\n"
                        "(or pass --skip-env-check to run anyway)"
                    )
                )
                sys.exit(2)

        if run_judge and (retrieval_only or summary_only):
            self.stderr.write(
                self.style.ERROR(
                    "--judge only applies to behavior cases (retrieval/summary cases "
                    "have no answer-vs-sources to judge; the summary suite runs its own "
                    "summary-fidelity judge automatically)."
                )
            )
            sys.exit(2)

        # Decide which suite(s) to run.
        if retrieval_only:
            suites = [("retrieval", RETRIEVAL_CASES_PATH, run_retrieval_case)]
        elif summary_only:
            suites = [("summary", SUMMARY_CASES_PATH, run_summary_case)]
        elif run_all:
            suites = [
                ("behavior", BEHAVIOR_CASES_PATH, run_behavior_case),
                ("retrieval", RETRIEVAL_CASES_PATH, run_retrieval_case),
                ("summary", SUMMARY_CASES_PATH, run_summary_case),
            ]
        else:
            suites = [("behavior", BEHAVIOR_CASES_PATH, run_behavior_case)]

        all_results: list[CaseResult] = []
        any_failed = False

        for label, path, runner in suites:
            try:
                cases = load_cases(path)
            except FileNotFoundError:
                self.stderr.write(self.style.ERROR(f"{label} cases not found at {path}"))
                sys.exit(2)
            except Exception as e:  # noqa: BLE001
                self.stderr.write(self.style.ERROR(f"Failed to parse {path}: {e}"))
                sys.exit(2)

            if case_id_filter:
                cases = [c for c in cases if c.get("id") == case_id_filter]
                if not cases:
                    # The case might live in a different suite; skip this one.
                    continue

            self.stdout.write(
                f"\n=== {label} suite ({len(cases)} case{'s' if len(cases) != 1 else ''}) ==="
            )
            for case in cases:
                result = runner(case)
                if run_judge and label == "behavior" and result.answer:
                    result.judge_scores = judge_answer(
                        query=result.query,
                        sources=result.sources,
                        answer=result.answer,
                        tool_results=result.tool_results,
                    )
                all_results.append(result)
                self._print_one(result)
                if not result.passed:
                    any_failed = True
                    if fail_fast:
                        self.stdout.write(self.style.WARNING("\n--fail-fast: stopping.\n"))
                        break
            else:
                # Inner loop completed without break — continue with the next suite.
                continue
            # Broke out of inner loop (fail-fast); stop running further suites too.
            break

        if case_id_filter and not all_results:
            self.stderr.write(self.style.ERROR(f"No case found with id={case_id_filter!r}"))
            sys.exit(2)

        self._print_summary(all_results, run_judge=run_judge)

        # Q0.4 — tool-selection north-star. The aggregate tool-selection
        # scalars are always reported against the floor; enforcement (exit 1
        # on a miss) is opt-in via --metric-gate (observe-only by default).
        gate_failed = self._check_metric_gate(all_results, enforce=metric_gate)

        if run_judge:
            self._persist_judge_run(all_results)

        if persist_metrics:
            self._persist_metrics_run(all_results)

        if any_failed or gate_failed:
            sys.exit(1)

    # Aggregate continuous metrics that have a published north-star floor
    # (SPOTLIGHT_QUALITY_ARCHITECTURE.md §4.5 / roadmap §1). Each maps to the
    # settings key holding its threshold.
    _NORTH_STAR_METRICS = {
        "tool_recall": "RAG_TOOL_SELECTION_NORTH_STAR",
        "tool_excl_ok": "RAG_TOOL_SELECTION_NORTH_STAR",
    }

    def _check_metric_gate(self, results: list[CaseResult], *, enforce: bool) -> bool:
        """Gate the aggregate tool-selection scalars against their north-star.

        Returns True when the gate FAILED (a metric mean is below its floor)
        and `enforce` is set — the caller turns that into exit 1. When
        `enforce` is False this only reports (observe-only). Metrics with no
        case population this run are skipped (you can't gate what you didn't
        measure).
        """
        from django.conf import settings  # noqa: PLC0415 — Django app-ready

        scored = [r for r in results if r.metrics]
        if not scored:
            return False

        violations: list[str] = []
        reported = False
        for metric, setting_key in self._NORTH_STAR_METRICS.items():
            vals = [r.metrics[metric] for r in scored if metric in r.metrics]
            if not vals:
                continue
            floor = float(settings.SEARCH_ENGINE.get(setting_key, 0.90))
            mean = sum(vals) / len(vals)
            ok = mean >= floor
            if not reported:
                self.stdout.write("\nNorth-star gate (Q0.4):")
                reported = True
            status = self.style.SUCCESS("OK  ") if ok else self.style.ERROR("FAIL")
            self.stdout.write(
                f"  {status} {metric}={mean:.3f} (n={len(vals)}) vs floor {floor:.2f}"
            )
            if not ok:
                violations.append(f"{metric}={mean:.3f} < {floor:.2f}")

        if violations:
            if enforce:
                self.stdout.write(
                    self.style.ERROR(
                        "  → north-star gate FAILED: " + "; ".join(violations)
                    )
                )
                return True
            self.stdout.write(
                self.style.WARNING(
                    "  → below north-star (observe-only; pass --metric-gate to enforce): "
                    + "; ".join(violations)
                )
            )
        return False

    def _index_mapping_drift(self) -> list[str]:
        """Compare the live index's text-field subfields against the
        canonical `build_mappings()`.

        Returns human-readable drift lines; [] = healthy. Additive-only
        check (live extras are fine — only MISSING subfields hurt, by
        silently disabling the query lanes search.py boosts). OpenSearch
        being unreachable is NOT drift — the suite will fail loudly on
        its own; we don't want the preflight masking that error shape.
        """
        try:
            from django.conf import settings  # noqa: PLC0415

            from origin.search_engine.index_config import build_mappings  # noqa: PLC0415
            from origin.search_engine.opensearch_client import get_client  # noqa: PLC0415

            alias = settings.SEARCH_ENGINE.get("OPENSEARCH_ALIAS", "knowledge_chunks_current")
            client = get_client()
            live = client.indices.get_mapping(index=alias)
            live_props = live[next(iter(live))]["mappings"].get("properties") or {}
        except Exception:  # noqa: BLE001 — connectivity is the suites' problem
            return []

        drift: list[str] = []
        for fname, spec in (build_mappings().get("properties") or {}).items():
            want = set((spec.get("fields") or {}).keys())
            if not want:
                continue
            have = set(((live_props.get(fname) or {}).get("fields")) or {})
            missing = sorted(want - have)
            if missing:
                drift.append(f"{fname}: live index is missing subfields {missing}")
        return drift

    def _print_one(self, r: CaseResult) -> None:
        if r.passed:
            label = self.style.SUCCESS("PASS")
        elif getattr(r, "infra_error", False):
            # Not-passed, but provider infrastructure died (429/5xx) —
            # excluded from continuous metrics; visually distinct so a
            # quota-weather night isn't read as a quality regression.
            label = self.style.WARNING("INFR")
        else:
            label = self.style.ERROR("FAIL")
        ttft_part = f", ttft {r.ttft_ms} ms" if getattr(r, "ttft_ms", -1) >= 0 else ""
        if r.tool_call_count > 0 or r.step_count > 0:
            detail = (
                f"({r.step_count} step{'s' if r.step_count != 1 else ''}, "
                f"{r.duration_ms} ms{ttft_part})"
            )
        else:
            detail = f"({r.duration_ms} ms{ttft_part})"
        self.stdout.write(f"  {label}  {r.case_id:<48} {detail}")
        for reason in r.failure_reasons:
            self.stdout.write(self.style.ERROR(f"        - {reason}"))
        if r.judge_scores is not None:
            j = r.judge_scores
            note = j.get("error") or j.get("notes") or ""
            # prose_faithfulness (D5) is nullable — None = the answer had
            # no link-form citations to score, shown as n/a, never 0.
            prose = j.get("prose_faithfulness")
            prose_part = f" prose={prose:.2f}" if prose is not None else " prose=n/a"
            self.stdout.write(
                "        "
                f"judge: faith={j.get('faithfulness', 0):.2f} "
                f"cite={j.get('citation_precision', 0):.2f} "
                f"compl={j.get('completeness', 0):.2f}"
                + prose_part
                + (f"  — {note}" if note else "")
            )
        if r.metrics:
            parts = "  ".join(f"{k}={v:.3f}" for k, v in sorted(r.metrics.items()))
            self.stdout.write(self.style.NOTICE(f"        metrics: {parts}"))

    def _print_summary(self, results: list[CaseResult], *, run_judge: bool) -> None:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        infra = [r.case_id for r in results if getattr(r, "infra_error", False)]
        infra_part = f" ({len(infra)} infra-errored, excluded from metrics)" if infra else ""
        self.stdout.write("")
        if passed == total:
            self.stdout.write(self.style.SUCCESS(f"{passed}/{total} passed."))
        else:
            self.stdout.write(self.style.ERROR(f"{passed}/{total} passed.{infra_part}"))
            failed = [r.case_id for r in results if not r.passed]
            self.stdout.write("Failures:")
            for cid in failed:
                marker = "  (infra)" if cid in infra else ""
                self.stdout.write(f"  - {cid}{marker}")

        if run_judge:
            judged = [r for r in results if r.judge_scores is not None]
            if judged:

                def _avg(key: str) -> float:
                    vals = [r.judge_scores.get(key, 0.0) for r in judged]
                    return sum(vals) / len(vals)

                # Nullable axis (D5): mean over the cases where it applies
                # (answers with link-form citations), n reported separately.
                prose_vals = [
                    r.judge_scores["prose_faithfulness"]
                    for r in judged
                    if r.judge_scores.get("prose_faithfulness") is not None
                ]
                prose_part = (
                    f"  prose={sum(prose_vals) / len(prose_vals):.2f} (n={len(prose_vals)})"
                    if prose_vals
                    else "  prose=n/a"
                )
                self.stdout.write(
                    self.style.NOTICE(
                        f"\nLLM judge ({len(judged)} cases): "
                        f"faith={_avg('faithfulness'):.2f}  "
                        f"cite={_avg('citation_precision'):.2f}  "
                        f"compl={_avg('completeness'):.2f}" + prose_part
                    )
                )

        # Continuous metrics (Q0) — retrieval rank quality (mrr /
        # recall_at_n, eval-mode fixture-based, NOT production recall) and
        # tool-selection (tool_recall / tool_excl_ok). Each metric is
        # averaged only over the cases that declared it, so denominators
        # differ — the per-metric (n=) makes that explicit.
        scored = [r for r in results if r.metrics]
        if scored:
            keys = sorted({k for r in scored for k in r.metrics})
            parts = []
            for k in keys:
                vals = [r.metrics[k] for r in scored if k in r.metrics]
                parts.append(f"{k}={sum(vals) / len(vals):.3f} (n={len(vals)})")
            self.stdout.write(self.style.NOTICE("\nContinuous metrics: " + "  ".join(parts)))

    def _run_basename(self) -> str:
        """`<timestamp>-<short-sha>` stem shared by persisted run files.

        Suppresses git stderr so "fatal: not a git repository" doesn't
        leak into eval output when CWD isn't a git checkout; falls back
        to `unknown` for the sha there.
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:  # noqa: BLE001 — outside git or no git installed
            sha = "unknown"
        return f"{ts}-{sha}"

    def _persist_metrics_run(self, results: list[CaseResult]) -> None:
        """Append one JSONL line per metric-bearing case to
        `agent/evals/runs/<ts>-metrics.jsonl` for trending retrieval
        quality (mrr / recall_at_n) across PRs. Only cases that declared
        rank-checkable gold are written; if none did, nothing is."""
        scored = [r for r in results if r.metrics]
        if not scored:
            self.stdout.write(
                self.style.WARNING(
                    "\n--persist-metrics: no case produced metrics (did the run "
                    "include the retrieval suite?); nothing written."
                )
            )
            return

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RUNS_DIR / f"{self._run_basename()}-metrics.jsonl"
        with out_path.open("w") as f:
            for r in scored:
                f.write(
                    json.dumps(
                        {
                            "case_id": r.case_id,
                            "passed": r.passed,
                            "duration_ms": r.duration_ms,
                            "metrics": r.metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self.stdout.write(
            self.style.SUCCESS(f"\nWrote metrics run to {out_path.relative_to(os.getcwd())}")
        )

    def _persist_judge_run(self, results: list[CaseResult]) -> None:
        """Append one JSONL line per case to `agent/evals/runs/<ts>.jsonl`.

        Includes the judge scores plus enough context (case id, query,
        answer, sources) to inspect or re-judge later. Useful for
        tracking quality trends across PRs.
        """
        judged = [r for r in results if r.judge_scores is not None]
        if not judged:
            return

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RUNS_DIR / f"{self._run_basename()}.jsonl"
        with out_path.open("w") as f:
            for r in judged:
                f.write(
                    json.dumps(
                        {
                            "case_id": r.case_id,
                            "passed": r.passed,
                            "duration_ms": r.duration_ms,
                            "ttft_ms": r.ttft_ms,
                            "query": r.query,
                            "answer": r.answer,
                            "sources": [
                                {
                                    "entity_id": s.get("entity_id"),
                                    "title": s.get("title"),
                                }
                                for s in r.sources
                            ],
                            "tool_results": [
                                {
                                    "tool_name": tr.get("tool_name"),
                                    "arguments": tr.get("arguments"),
                                    "result": tr.get("result"),
                                }
                                for tr in r.tool_results
                            ],
                            "judge": r.judge_scores,
                            "metrics": r.metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self.stdout.write(
            self.style.SUCCESS(f"\nWrote judge run to {out_path.relative_to(os.getcwd())}")
        )
