"""Tool-trajectory capture + tolerant diff for behavior cases.

SPOTLIGHT_AGENT_CHANGE_SAFETY.md §5.1 (genos-docs): when a PR changes
agent logic or tools, the reviewer should see WHICH TOOLS the agent now
reaches for, per eval case — the behavioral delta that the pass/fail
assertions and judge scores don't directly surface.

Tolerance is the design constraint. The model legitimately varies
run-to-run (fires a tool once vs twice, reorders calls), so an exact
call-sequence snapshot would be perpetually red and get rubber-stamped.
Each case therefore collapses to the SET of distinct tool names that
actually executed (from the controller `trace_hook` captures on
`CaseResult.tool_results`), and only set membership is diffed.

Report-only by contract: the CI workflow surfaces the diff to the
reviewer (job summary / PR comment) and never gates on it — some churn
is normal model variance and needs a human read.

The committed baseline (`trajectory_baseline.json`, next to this file)
is regenerated deliberately via
`manage.py agent_eval_trajectory --write-baseline` when a trajectory
change is intentional — the baseline update then shows up in the same
PR diff, reviewable like any golden file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "trajectory_baseline.json"
BASELINE_VERSION = 1


def tool_set(result) -> list[str]:
    """Sorted distinct tool names that executed for one CaseResult.

    Uses `tool_results` (controller trace_hook captures) rather than raw
    NDJSON events: a trace entry means the tool actually ran. Repeat
    calls collapse — that's the tolerance.
    """
    return sorted(
        {
            trace.get("tool_name")
            for trace in (result.tool_results or [])
            if trace.get("tool_name")
        }
    )


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, list[str]] | None:
    """The committed baseline, or None when it hasn't been generated yet.

    None is a supported state, not an error — the diff consumer prints a
    bootstrap hint instead of failing, so the feature can land before
    the first baseline is committed.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != BASELINE_VERSION:
        raise ValueError(
            f"{path.name} has version {version!r}; this code expects "
            f"{BASELINE_VERSION}. Regenerate with "
            "`manage.py agent_eval_trajectory --write-baseline`."
        )
    return {case_id: sorted(tools) for case_id, tools in (data.get("cases") or {}).items()}


def dump_baseline(cases: dict[str, list[str]], path: Path = BASELINE_PATH) -> None:
    payload = {
        "version": BASELINE_VERSION,
        # Informational only — deliberately NOT part of the diff, so
        # regenerating an identical baseline still produces a one-line
        # change at most.
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": {case_id: sorted(tools) for case_id, tools in sorted(cases.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class TrajectoryDiff:
    # case_id -> (tools added vs baseline, tools removed vs baseline)
    changed: dict[str, tuple[list[str], list[str]]] = field(default_factory=dict)
    new_cases: list[str] = field(default_factory=list)  # in run, not in baseline
    missing_cases: list[str] = field(default_factory=list)  # in baseline, not in run
    unchanged: int = 0

    @property
    def is_clean(self) -> bool:
        return not (self.changed or self.new_cases or self.missing_cases)


def diff_trajectories(
    baseline: dict[str, list[str]], current: dict[str, list[str]]
) -> TrajectoryDiff:
    changed: dict[str, tuple[list[str], list[str]]] = {}
    shared = set(baseline) & set(current)
    for case_id in sorted(shared):
        added = sorted(set(current[case_id]) - set(baseline[case_id]))
        removed = sorted(set(baseline[case_id]) - set(current[case_id]))
        if added or removed:
            changed[case_id] = (added, removed)
    return TrajectoryDiff(
        changed=changed,
        new_cases=sorted(set(current) - set(baseline)),
        missing_cases=sorted(set(baseline) - set(current)),
        unchanged=len(shared) - len(changed),
    )


def format_diff(diff: TrajectoryDiff) -> str:
    """Markdown rendering — used verbatim for stdout, the CI job summary,
    and the best-effort PR comment."""
    lines = [
        "## Agent tool-trajectory diff (report-only)",
        "",
        f"- unchanged: **{diff.unchanged}** case(s)",
        f"- changed: **{len(diff.changed)}** case(s)",
        f"- new cases (no baseline entry): **{len(diff.new_cases)}**",
        f"- baseline cases missing from this run: **{len(diff.missing_cases)}**",
    ]
    if diff.changed:
        lines += [
            "",
            "| case | tools added | tools removed |",
            "|---|---|---|",
        ]
        for case_id, (added, removed) in diff.changed.items():
            plus = ", ".join(f"+{t}" for t in added) or "—"
            minus = ", ".join(f"-{t}" for t in removed) or "—"
            lines.append(f"| `{case_id}` | {plus} | {minus} |")
    if diff.new_cases:
        lines += ["", "New cases: " + ", ".join(f"`{c}`" for c in diff.new_cases)]
    if diff.missing_cases:
        lines += ["", "Missing cases: " + ", ".join(f"`{c}`" for c in diff.missing_cases)]
    lines += [
        "",
        "_Set-of-tools diff (repeat calls collapsed). Some churn is normal "
        "model variance — read as a review signal, not a failure. To accept "
        "an intentional change: `manage.py agent_eval_trajectory "
        "--write-baseline` and commit the updated baseline._",
    ]
    return "\n".join(lines)
