"""`python manage.py agent_eval_trajectory` — which tools fire, per case.

Runs the behavior suite (judge-less — roughly half the LLM spend of
`agent_eval --judge`) and reduces each case to the set of tool names
that executed. Three uses:

    # Capture + diff against the committed baseline (default; report-only)
    python manage.py agent_eval_trajectory

    # Accept an intentional trajectory change (then commit the file)
    python manage.py agent_eval_trajectory --write-baseline

    # One case while iterating locally
    python manage.py agent_eval_trajectory --case simple_rag_wip_tasks

Exit code is 0 even when trajectories changed — the diff is a review
signal, never a gate (SPOTLIGHT_AGENT_CHANGE_SAFETY.md §5.1). Pass
`--strict` to exit 1 on any diff (local pre-commit use only; do NOT
wire into CI as a gate — model variance would flap it red).

Needs the same environment as `agent_eval`: seeded fixture
(`agent_eval_setup --reseed`), OpenSearch, and LLM credentials.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from origin.search_engine.agent.evals.runner import load_cases, run_behavior_case
from origin.search_engine.agent.evals.trajectory import (
    BASELINE_PATH,
    diff_trajectories,
    dump_baseline,
    format_diff,
    load_baseline,
    tool_set,
)


class Command(BaseCommand):
    help = "Capture per-case agent tool trajectories and diff them against the committed baseline."

    def add_arguments(self, parser):
        parser.add_argument(
            "--case",
            help="Run only the case with this id (local iteration).",
        )
        parser.add_argument(
            "--write-baseline",
            action="store_true",
            help=f"Regenerate {BASELINE_PATH.name} from this run instead of diffing.",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Exit 1 on any trajectory change (local use; never a CI gate).",
        )

    def handle(self, *args, **opts):
        cases = load_cases()
        if opts.get("case"):
            cases = [c for c in cases if c.get("id") == opts["case"]]
            if not cases:
                self.stderr.write(self.style.ERROR(f"no case with id {opts['case']!r}"))
                raise SystemExit(2)

        current: dict[str, list[str]] = {}
        for case in cases:
            result = run_behavior_case(case)
            tools = tool_set(result)
            current[result.case_id] = tools
            marker = "PASS" if result.passed else "FAIL"
            self.stdout.write(
                f"[{marker}] {result.case_id}: {', '.join(tools) or '(no tools called)'}"
            )
            # A failed case's (possibly truncated) trajectory is still
            # captured — a case that stopped calling tools because it now
            # crashes IS a behavioral delta the reviewer should see.

        if opts["write_baseline"]:
            if opts.get("case"):
                # A partial run must never overwrite the full baseline.
                self.stderr.write(
                    self.style.ERROR("--write-baseline cannot be combined with --case")
                )
                raise SystemExit(2)
            dump_baseline(current)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Baseline written: {BASELINE_PATH} ({len(current)} cases). "
                    "Commit it so CI diffs against this run."
                )
            )
            return

        baseline = load_baseline()
        if baseline is None:
            self.stdout.write(
                self.style.WARNING(
                    f"No committed baseline ({BASELINE_PATH.name}) — nothing to "
                    "diff. Bootstrap once with --write-baseline and commit the file."
                )
            )
            return
        if opts.get("case"):
            # Diff just the selected slice so single-case iteration works.
            baseline = {k: v for k, v in baseline.items() if k in current}

        diff = diff_trajectories(baseline, current)
        self.stdout.write("")
        self.stdout.write(format_diff(diff))
        if opts["strict"] and not diff.is_clean:
            raise SystemExit(1)
