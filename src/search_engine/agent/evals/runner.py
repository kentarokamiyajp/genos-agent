"""Eval-harness runner.

Two modes, sharing the same `CaseResult` dataclass and CLI:

  1. **Behavior mode** — case file `cases.yaml`. Each case runs the
     full agent loop (`run_agent`) and asserts on the emitted NDJSON
     event stream (which tools were called, what the answer says,
     etc.). Used to catch regressions in agent decision-making.

  2. **Retrieval mode** — case file `retrieval_cases.yaml`. Each case
     calls `search(...)` directly and asserts on the ranked entity
     list (gold-standard recall checks: "query X must return entity Y
     in top N"). No LLM calls; fast and free.

Design notes:

  * Both modes return the same `CaseResult` shape so the CLI prints
    them identically.
  * Behavior cases may seed an adversarial note via
    `setup.inject_note`; retrieval cases don't currently need a
    `setup` block (gold data is already in the index).
  * Assertions are declarative (in YAML) rather than imperative
    Python so case authors don't have to read the runner internals.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from origin.search_engine.agent.abstention import is_abstention
from origin.search_engine.agent.controller import _irrelevant_tool_families, run_agent
from origin.search_engine.agent.tools import REGISTRY, ToolContext
from origin.search_engine.search import search

log = logging.getLogger(__name__)

BEHAVIOR_CASES_PATH = Path(__file__).parent / "cases.yaml"
RETRIEVAL_CASES_PATH = Path(__file__).parent / "retrieval_cases.yaml"
SUMMARY_CASES_PATH = Path(__file__).parent / "summary_cases.yaml"

# Kept for backwards compatibility — callers that import `CASES_PATH`
# get the behavior path (the original meaning).
CASES_PATH = BEHAVIOR_CASES_PATH


def _resolve_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """If `case.fixture == True`, fill in team_id / user_id from the
    deterministic eval fixture (see `agent/evals/fixture.py`).

    Mutates and returns the same dict for convenience. Cases that
    pin their own team_id are left untouched — useful for legacy
    dev-DB fixture cases and adversarial cross-tenant tests.
    """
    if not case.get("fixture"):
        return case

    # Lazy import — the fixture module touches Django models, which
    # would blow up if imported at module load before app-ready.
    from origin.search_engine.agent.evals.fixture import (  # noqa: PLC0415
        FIXTURE_USER_ID,
        ensure_fixture,
    )

    info = ensure_fixture()
    case.setdefault("team_id", info["team_id"])
    case.setdefault("user_id", str(FIXTURE_USER_ID))
    return case


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    duration_ms: int
    failure_reasons: list[str] = field(default_factory=list)
    # Populated for behavior cases; meaningless for retrieval cases.
    step_count: int = 0
    tool_call_count: int = 0
    # Captured behavior-case artefacts — populated only when the caller
    # asked for them (e.g. `judge=True`). Kept on the result dataclass
    # so the LLM judge / trace writer can read them without re-running.
    query: str = ""
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    # Full tool-call traces captured via the controller's `trace_hook`.
    # Each entry: {"tool_name": str, "arguments": dict, "result": dict}.
    # Used by the LLM judge to verify the answer's factual claims
    # against the actual data the model saw (sources alone are too
    # sparse for structured-tool answers — they carry only entity_id
    # and title, not the status/due_date/priority the model legitimately
    # quotes from a `list_tasks` result).
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    # Phase 4.4 — milliseconds from run_agent invocation to the first
    # `answer_delta` event (the strict TTFT metric). -1 if no
    # answer_delta was ever emitted (model errored or returned no text).
    ttft_ms: int = -1
    # The run died on an LLM-provider infrastructure error (Vertex 429
    # quota / 5xx) even after one retry. Counted as not-passed but
    # EXCLUDED from continuous metrics and reported separately — infra
    # weather must not read as an agent-quality regression.
    infra_error: bool = False
    # Optional LLM-judge scores; only set when `--judge` was on.
    judge_scores: dict[str, Any] | None = None
    # Continuous quality metrics layered on top of the binary pass/fail
    # (Q0 of SPOTLIGHT_QUALITY_ARCHITECTURE.md). Retrieval cases populate
    # rank-based signals (`mrr`, `recall_at_n`); behavior cases leave it
    # empty today (see `_retrieval_metrics` for why tool-selection is not
    # yet a continuous metric on this suite). Empty `{}` → no metric for
    # this case, so aggregators skip it.
    metrics: dict[str, float] = field(default_factory=dict)


def load_cases(path: Path = BEHAVIOR_CASES_PATH) -> list[dict[str, Any]]:
    """Read and parse a YAML cases file. Raises on missing/invalid."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a top-level YAML list of cases; got {type(data).__name__}"
        )
    return data


# --------------------------------------------------------------------------- #
# Behavior mode (Phase 4 — agent-loop assertions)                             #
# --------------------------------------------------------------------------- #


# Seconds to wait before the single infra retry — long enough for a
# transient Vertex 429 burst to clear, short enough not to blow up a
# 131-case suite run when quota is genuinely gone.
INFRA_RETRY_SLEEP_S = 20

# Signatures of LLM-provider infrastructure failures in fatal `error`
# events (the controller prefixes them "LLM call failed: ..."). Matched
# only inside that prefix so an agent ANSWER mentioning "quota" can
# never trip it.
_INFRA_SIGNATURES = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "503",
    "unavailable",
    "deadline exceeded",
)


def _infra_failure(events: list[dict[str, Any]]) -> bool:
    """True when the run died on provider infrastructure, not behavior."""
    for e in events:
        if e.get("type") != "error":
            continue
        msg = (e.get("message") or "").lower()
        if "llm call failed" in msg and any(s in msg for s in _INFRA_SIGNATURES):
            return True
    return False


def run_behavior_case(case: dict[str, Any]) -> CaseResult:
    """Execute one behavior case through the full agent loop.

    Single-turn shape (default):
        - id: ...
          query: "..."
          expect: {...}

    Multi-turn shape (Phase 3.5):
        - id: ...
          turns:
            - query: "first turn"           # no expect
            - query: "second turn"          # no expect
            - query: "final turn"
              expect: {...}                  # assertions on the final turn

        Assertions live on the LAST turn (or in the case's top-level
        `expect` block). The runner threads `prior_turns` between turns
        the same way `agent_views.py` does in production, so the case
        exercises real multi-turn memory.
    """
    case = _resolve_fixture(case)
    case_id = case.get("id") or "(unnamed)"

    # Multi-turn fork — handled in a dedicated helper that mirrors the
    # single-turn flow per-turn and thread prior turns between them.
    if "turns" in case:
        return _run_multiturn_case(case, case_id)

    query = case.get("query") or ""
    team_id = case.get("team_id") or ""
    user_id = case.get("user_id") or ""
    expect = case.get("expect") or {}
    setup = case.get("setup") or {}

    if not query or not team_id or not user_id:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=0,
            failure_reasons=["case is missing query/team_id/user_id"],
        )

    cleanup_handles: list[Any] = []
    started = time.monotonic()
    try:
        if "inject_note" in setup:
            handle = _setup_inject_note(setup["inject_note"], team_id=team_id, user_id=user_id)
            cleanup_handles.append(handle)
        if "seed_conversation" in setup:
            handle = _setup_seed_conversation(
                setup["seed_conversation"], team_id=team_id, user_id=user_id
            )
            cleanup_handles.append(handle)

        ctx = ToolContext(team_id=team_id, user_id=user_id)

        def _attempt() -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None]:
            """One full agent run with event / trace / TTFT capture.

            Phase 4.4 — wrapping the emit callback is the smallest
            non-invasive timing hook (controller code untouched);
            timestamps are relative to the run_agent invocation.
            """
            attempt_events: list[dict[str, Any]] = []
            attempt_traces: list[dict[str, Any]] = []
            emit_t0 = time.monotonic()
            ttft: float | None = None

            def _ts_emit(event: dict[str, Any]) -> None:
                nonlocal ttft
                if (
                    ttft is None
                    and event.get("type") == "answer_delta"
                    and (event.get("text") or "")
                ):
                    ttft = time.monotonic() - emit_t0
                attempt_events.append(event)

            def _capture_trace(
                name: str, args: dict[str, Any], result: dict[str, Any]
            ) -> None:
                attempt_traces.append({"tool_name": name, "arguments": args, "result": result})

            run_agent(query, ctx, _ts_emit, run_id=None, trace_hook=_capture_trace)
            return attempt_events, attempt_traces, ttft

        try:
            events, tool_traces, ttft_s = _attempt()
            # LLM-provider infrastructure failure (Vertex 429 quota /
            # 5xx): not an agent-quality signal. Retry once after a
            # breather — transient quota noise usually clears. If it
            # persists, the case is flagged `infra_error` and EXCLUDED
            # from continuous metrics, so e.g. the tool_recall
            # north-star can't be breached by quota weather (exactly
            # what happened on the 2026-07-09 nightly: two 429'd cases
            # scored tool_recall=0.0 and "breached" the floor).
            if _infra_failure(events):
                time.sleep(INFRA_RETRY_SLEEP_S)
                events, tool_traces, ttft_s = _attempt()
        except Exception as e:  # noqa: BLE001 — report as failure rather than crash the suite
            duration_ms = int((time.monotonic() - started) * 1000)
            return CaseResult(
                case_id=case_id,
                passed=False,
                duration_ms=duration_ms,
                failure_reasons=[f"run_agent crashed: {e!r}"],
            )

        infra = _infra_failure(events)
        reasons = _check_behavior_expectations(events, expect)
        if infra:
            reasons = ["infra: LLM-provider failure after retry (excluded from metrics)"] + reasons
        duration_ms = int((time.monotonic() - started) * 1000)

        tool_calls = [e for e in events if e.get("type") == "tool_call_start"]
        step_count = max((e.get("step", -1) for e in tool_calls), default=-1) + 1

        # Capture answer + last `sources` snapshot so the LLM judge
        # (or any post-hoc analyser) can score this run without
        # re-executing it.
        answer_text = "".join(
            e.get("text") or "" for e in events if e.get("type") == "answer_delta"
        )
        source_events = [e for e in events if e.get("type") == "sources"]
        last_sources = source_events[-1].get("sources", []) if source_events else []

        return CaseResult(
            case_id=case_id,
            passed=not reasons,
            duration_ms=duration_ms,
            failure_reasons=reasons,
            step_count=step_count,
            tool_call_count=len(tool_calls),
            query=query,
            answer=answer_text,
            sources=list(last_sources),
            tool_results=tool_traces,
            ttft_ms=int(ttft_s * 1000) if ttft_s is not None else -1,
            infra_error=infra,
            # Infra-errored runs carry NO metrics: a 429'd run's
            # tool_recall=0 is noise, not signal, and would drag the
            # north-star aggregate below its floor.
            metrics={}
            if infra
            else {
                **_tool_selection_metrics(events, expect),
                **_abstention_metric(events, expect),
                **_citation_style_metric(events, expect),
                **_surface_metric(query),
            },
        )
    finally:
        for handle in cleanup_handles:
            try:
                handle()
            except Exception:  # noqa: BLE001
                log.exception("Cleanup handle failed for case %s", case_id)


# Backwards-compatible alias. The Phase-4 management command imported
# this name; keep it so existing call sites don't break.
run_case = run_behavior_case


def _run_multiturn_case(case: dict[str, Any], case_id: str) -> CaseResult:
    """Phase 3.5 — execute a multi-turn case by running the agent loop
    once per turn, threading prior (query, answer) pairs between turns.

    Mirrors the production agent_views.py session-threading shape so the
    multi-turn-memory mechanism is exercised end-to-end. The eval does
    NOT honor `SESSION_MAX_PRIOR_TURNS` truncation itself — it passes
    ALL prior turns to `run_agent`, the same way production would for a
    fresh session. Truncation/summarisation is applied INSIDE the
    prior-turns prep helper, so flag-toggling `RAG_SESSION_ROLLING_SUMMARY`
    in an A/B works without changing the case shape.

    Assertions live on the LAST turn's expectations (or on the case's
    top-level `expect`). Earlier turns are setup; we never fail-fast on
    a per-turn intermediate.
    """
    turns = case.get("turns") or []
    if not turns:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=0,
            failure_reasons=["multi-turn case has empty `turns` list"],
        )

    team_id = case.get("team_id") or ""
    user_id = case.get("user_id") or ""
    if not team_id or not user_id:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=0,
            failure_reasons=["case is missing team_id/user_id"],
        )

    # The terminal expectation set: prefer the last turn's `expect`,
    # else fall back to the case-level `expect`.
    final_turn = turns[-1]
    final_expect = final_turn.get("expect") or case.get("expect") or {}

    # Phase 3.5 — defer truncation + (optional) rolling summary to the
    # shared helper, so a single `RAG_SESSION_ROLLING_SUMMARY=true`
    # override in agent_eval_compare exercises the exact code path
    # production /ask/ uses. Helper is no-op when the flag is off OR
    # the session is shorter than the verbatim window — i.e. matches
    # pre-3.5 behavior unless explicitly opted in.
    from origin.search_engine.agent.multi_turn import build_prior_context  # noqa: PLC0415

    ctx = ToolContext(team_id=team_id, user_id=user_id)
    prior_turns: list[tuple[str, str]] = []
    last_events: list[dict[str, Any]] = []
    last_tool_traces: list[dict[str, Any]] = []
    last_query = ""
    last_ttft_s: float | None = None
    started = time.monotonic()

    for i, turn in enumerate(turns):
        q = (turn.get("query") or "").strip()
        if not q:
            return CaseResult(
                case_id=case_id,
                passed=False,
                duration_ms=int((time.monotonic() - started) * 1000),
                failure_reasons=[f"turn {i + 1} is missing a non-empty `query`"],
            )

        events: list[dict[str, Any]] = []
        tool_traces: list[dict[str, Any]] = []
        # Per-turn TTFT — we only retain the FINAL turn's value below.
        per_turn_t0 = time.monotonic()
        per_turn_ttft_s: float | None = None

        def _ts_emit(event: dict[str, Any]) -> None:
            nonlocal per_turn_ttft_s
            if (
                per_turn_ttft_s is None
                and event.get("type") == "answer_delta"
                and (event.get("text") or "")
            ):
                per_turn_ttft_s = time.monotonic() - per_turn_t0
            events.append(event)

        def _capture_trace(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
            tool_traces.append({"tool_name": name, "arguments": args, "result": result})

        verbatim_turns, summary = build_prior_context(prior_turns)
        try:
            run_agent(
                q,
                ctx,
                _ts_emit,
                run_id=None,
                prior_turns=verbatim_turns,
                prior_summary=summary,
                trace_hook=_capture_trace,
            )
        except Exception as e:  # noqa: BLE001
            return CaseResult(
                case_id=case_id,
                passed=False,
                duration_ms=int((time.monotonic() - started) * 1000),
                failure_reasons=[f"turn {i + 1} run_agent crashed: {e!r}"],
            )

        answer_text = "".join(
            (e.get("text") or "") for e in events if e.get("type") == "answer_delta"
        )
        prior_turns.append((q, answer_text))
        last_query = q
        last_events = events
        last_tool_traces = tool_traces
        last_ttft_s = per_turn_ttft_s

    # Score only the final turn. Infra detection but no retry here —
    # replaying every prior turn to retry the last one would multiply
    # LLM spend; flagging + excluding from metrics is the part that
    # protects the aggregates.
    infra = _infra_failure(last_events)
    reasons = _check_behavior_expectations(last_events, final_expect)
    if infra:
        reasons = ["infra: LLM-provider failure (excluded from metrics)"] + reasons
    duration_ms = int((time.monotonic() - started) * 1000)

    tool_calls = [e for e in last_events if e.get("type") == "tool_call_start"]
    step_count = max((e.get("step", -1) for e in tool_calls), default=-1) + 1
    answer_text = "".join(
        (e.get("text") or "") for e in last_events if e.get("type") == "answer_delta"
    )
    source_events = [e for e in last_events if e.get("type") == "sources"]
    last_sources = source_events[-1].get("sources", []) if source_events else []

    return CaseResult(
        case_id=case_id,
        passed=not reasons,
        duration_ms=duration_ms,
        failure_reasons=reasons,
        step_count=step_count,
        tool_call_count=len(tool_calls),
        query=last_query,
        answer=answer_text,
        sources=list(last_sources),
        tool_results=last_tool_traces,
        ttft_ms=int(last_ttft_s * 1000) if last_ttft_s is not None else -1,
        infra_error=infra,
        metrics={}
        if infra
        else {
            **_tool_selection_metrics(last_events, final_expect),
            **_abstention_metric(last_events, final_expect),
            **_citation_style_metric(last_events, final_expect),
            **_surface_metric(last_query),
        },
    )


# `_CITATION_RE` finds entity-id references in answer text, in BOTH forms
# the agent emits (§4.6 D5): the natural-prose link `[prose](type:id)` (id in
# group 1, link alternative FIRST so it consumes the whole `[label](id)`) and
# the bare `[type:id]` fallback (id in group 2). The id pattern matches what
# the agent uses: `chat:pm:1:thread:3`, `task:42`, `note:personal:7`, etc.
# Keep in sync with `_INLINE_CITATION_RE` in controller.py.
_CITATION_RE = re.compile(r"\[[^\]]*\]\(([a-z][a-z0-9_:\-]+)\)|\[([a-z][a-z0-9_:\-]+)\]")

# Bare-form-only counterpart of `judge.CITATION_LINK_RE`, for the D5
# adoption metric below: the bracket content must itself be a typed
# entity id (a link's prose label never matches). Same entity-type
# vocabulary as the link regex — ordinary bracketed text ("[sic]",
# "[reminder: …]") never counts.
_CITATION_BARE_RE = re.compile(
    r"\[((?:chat|task|note|project|todo|milestone)(?::[a-z0-9_\-]+)+)\](?!\()"
)

# A link LABEL that is itself a raw entity token — `[task:42](task:42)`.
# Weak models (seen with gemini-flash) emit this instead of natural
# prose; the frontend has to repair it, and it reads as a raw id in any
# other consumer of the answer text.
_RAW_TOKEN_LABEL_RE = re.compile(r"^(?:chat|task|note|project|todo|milestone):\S+$")

# Per-type id-shape vocabulary for `citation_wellformed_rate`. Matches
# every id form the system prompt teaches and the chunkers emit —
# anything else (e.g. gemini-flash's invented `chat:…:msg:<uuid>`
# segment, copied from tool results) is malformed: it resolves to no
# source and would surface as raw text without frontend repair. Keep in
# sync with the prompt's citation section and the frontend's
# CITATION_PATTERN vocabulary.
_WELLFORMED_ID_RES = {
    "task": re.compile(r"^task:\d+$"),
    "project": re.compile(r"^project:\d+$"),
    "milestone": re.compile(r"^milestone:\d+$"),
    "note": re.compile(r"^note:(?:personal|my|task|chat):\d+$"),
    "todo": re.compile(r"^todo:\d{4}-\d{2}-\d{2}(?::item:\d+)?$"),
    # chat ids are v3 UUIDs today but legacy ints still exist in old
    # chunks — accept any [a-z0-9-] segment. The tail may ONLY be a
    # `:thread:` segment; `:msg:` (or anything else) is malformed.
    "chat": re.compile(r"^chat:(?:dm|gm|pm|mdm):[a-z0-9\-]+(?::thread:[a-z0-9\-]+)?$"),
}


def _citation_id_wellformed(cited_id: str) -> bool:
    etype = cited_id.split(":", 1)[0]
    pattern = _WELLFORMED_ID_RES.get(etype)
    return bool(pattern and pattern.match(cited_id))


def _citation_style_metric(events: list[dict[str, Any]], expect: dict[str, Any]) -> dict[str, float]:
    """D5 prose-citation adoption rate (§4.6).

    `prose_citation_rate` = link-form citations / all citations — how
    often the model cites in the natural-prose `[prose](type:id)` form
    the D5 prompt asks for, vs falling back to the bare `[type:id]`
    token (which the frontend strips to a chip). This is the ADOPTION
    half of D5 measurement; the QUALITY half (is the prose truthful?)
    is the judge's nullable `prose_faithfulness` axis.

    Skip-when-immovable: emitted only for cases that positively expect
    citations (`has_citations` / `citations_contain` /
    `citations_count_at_least`) — on any other case the metric can't
    move and would just dilute the aggregate. 0.0 means "citations
    expected, none emitted in link form" (including the none-at-all
    case, which the binary assertion already fails separately).
    """
    # `citation_style_metric: true` is a METRIC-ONLY opt-in (precedent:
    # the `should_abstain` metric gold): it feeds these style metrics
    # without adding a binary citation assertion. Use it on cases where
    # a weak model only *sometimes* cites — `has_citations: true` there
    # would flake the deploy-gating suite, but the style of whatever
    # citations do appear is still worth measuring.
    positively_expects = any(
        k in expect
        for k in (
            "has_citations",
            "citations_contain",
            "citations_count_at_least",
            "citation_style_metric",
        )
    ) and not expect.get("no_citations")
    if not positively_expects:
        return {}
    from origin.search_engine.agent.evals.judge import extract_prose_citations  # noqa: PLC0415

    answer = "".join((e.get("text") or "") for e in events if e.get("type") == "answer_delta")
    links = extract_prose_citations(answer)
    bare_ids = _CITATION_BARE_RE.findall(answer)
    total = len(links) + len(bare_ids)
    if total == 0:
        return {"prose_citation_rate": 0.0}

    # `citation_wellformed_rate` — the STYLE half the adoption rate is
    # blind to. A citation is well-formed when (a) a link's label is
    # prose, not the raw token itself (`[task:42](task:42)` counts as
    # "link-form" for adoption but is malformed here), and (b) the cited
    # id matches the taught per-type shape (catches invented segments
    # like `chat:…:msg:<uuid>`). Both malformations were observed
    # verbatim from gemini-flash (2026-07-05); without frontend repair
    # they render as raw ids. 1.0 = every citation clean.
    wellformed = sum(
        1
        for label, cited_id in links
        if not _RAW_TOKEN_LABEL_RE.match(label.strip()) and _citation_id_wellformed(cited_id)
    ) + sum(1 for cited_id in bare_ids if _citation_id_wellformed(cited_id))
    return {
        "prose_citation_rate": len(links) / total,
        "citation_wellformed_rate": wellformed / total,
    }


def _abstention_metric(events: list[dict[str, Any]], expect: dict[str, Any]) -> dict[str, float]:
    """Continuous abstention-correctness signal (Q0, §4.1).

    Reads the metric-only gold `should_abstain` (true/false). `abstention_correct`
    is 1.0 when the answer's abstention status matches the gold, else 0.0 —
    so it catches BOTH error directions: a false answer on an unanswerable
    query (should_abstain: true, but answered) AND a false abstention on an
    answerable one, incl. the empty-structured-result case ("no overdue
    tasks" is an answer, should_abstain: false). Non-gating; `{}` when the
    case doesn't declare `should_abstain`.
    """
    if "should_abstain" not in expect:
        return {}
    answer = "".join((e.get("text") or "") for e in events if e.get("type") == "answer_delta")
    correct = is_abstention(answer) == bool(expect["should_abstain"])
    return {"abstention_correct": 1.0 if correct else 0.0}


def _surface_metric(query: str) -> dict[str, float]:
    """Deterministic tool-surface size under RAG_TOOL_SUBSETTING (§4.5).

    `tools_declared` = how many tools the model would see for this query.
    A SURFACE / COST signal, NOT a quality score: the A/B diff shows it
    shrink when subsetting is on. Always emitted so both runs carry it for
    a clean delta. The tool-selection-*quality* benefit is a hypothesis to
    validate on production (the F2 judge sampler), not on this fixture
    suite — no case here exercises the peripheral families subsetting
    drops, so the suite only confirms no-regression.
    """
    from django.conf import settings  # noqa: PLC0415 — lazy: Django app-ready

    excluded = (
        _irrelevant_tool_families(query)
        if settings.SEARCH_ENGINE.get("RAG_TOOL_SUBSETTING", False)
        else set()
    )
    return {"tools_declared": float(len(REGISTRY) - len(excluded))}


def _tool_selection_metrics(
    events: list[dict[str, Any]], expect: dict[str, Any]
) -> dict[str, float]:
    """Continuous tool-selection signals (Q0) layered on the binary tool
    assertions.

    Reads ONLY the metric-only gold fields — `expected_tools` and
    `forbidden_tools` — never the gating `tools_used_contains` /
    `tools_used_excludes`. That's deliberate: a gating assertion is
    structurally pinned (a *passing* `tools_used_contains` case always has
    recall 1.0; a passing `tools_used_excludes` case always has excl_ok
    1.0), so reading it would inject constant 1.0s that swamp the real
    signal — the same binary-in-a-costume trap as recall@n on singleton
    gold. The metric-only fields can actually move:

      * tool_recall  = |expected_tools ∩ used| / |expected_tools|
        Fractional when the model takes a one-tool shortcut on a
        genuinely multi-tool question.
      * tool_excl_ok = |forbidden_tools \\ used| / |forbidden_tools|
        Drops when the agent over-reaches to a tool it shouldn't (e.g.
        paid web search on an internal question).

    Both are NON-gating — the case's pass/fail comes from its other
    assertions — so path-sensitive multi-tool / negative gold can't
    flaky-fail the CI gate. See the `expected_tools` / `forbidden_tools`
    cases in cases.yaml. Returns `{}` when neither field is declared.
    """
    used = {e.get("tool_name") for e in events if e.get("type") == "tool_call_start"}
    out: dict[str, float] = {}
    expected = expect.get("expected_tools") or []
    if expected:
        req = set(expected)
        out["tool_recall"] = round(len(req & used) / len(req), 4)
    forbidden = expect.get("forbidden_tools") or []
    if forbidden:
        forb = set(forbidden)
        out["tool_excl_ok"] = round(len(forb - used) / len(forb), 4)
    return out


def _check_behavior_expectations(
    events: list[dict[str, Any]], expect: dict[str, Any]
) -> list[str]:
    """Run each declared assertion. Returns the list of failure reasons."""
    reasons: list[str] = []

    tools_used = [e.get("tool_name") for e in events if e.get("type") == "tool_call_start"]
    tool_call_count = len(tools_used)
    # Write-tool PROPOSALS never emit `tool_call_start` pre-approval —
    # the run pauses on `tool_call_pending_approval` instead — so
    # asserting "the model proposed write tool X" needs its own event
    # set (`pending_tools_contains` below).
    tools_pending = [
        e.get("tool_name") for e in events if e.get("type") == "tool_call_pending_approval"
    ]
    tool_errors = [e for e in events if e.get("type") == "tool_call_error"]
    fatal_errors = [e for e in events if e.get("type") == "error"]
    answer = "".join(e.get("text") or "" for e in events if e.get("type") == "answer_delta")
    citations_seen = {
        (m.group(1) or m.group(2)).lower() for m in _CITATION_RE.finditer(answer.lower())
    }
    step_count = (
        max(
            (e.get("step", -1) for e in events if "step" in e),
            default=-1,
        )
        + 1
    )

    def _add(reason: str) -> None:
        reasons.append(reason)

    if "tool_calls_at_least" in expect:
        n = int(expect["tool_calls_at_least"])
        if tool_call_count < n:
            _add(f"tool_calls_at_least: got {tool_call_count}, expected >= {n}")

    if "tool_calls_at_most" in expect:
        n = int(expect["tool_calls_at_most"])
        if tool_call_count > n:
            _add(f"tool_calls_at_most: got {tool_call_count}, expected <= {n}")

    if "tools_used_contains" in expect:
        required = set(expect["tools_used_contains"])
        seen = set(tools_used)
        missing = required - seen
        if missing:
            _add(f"tools_used_contains: missing {sorted(missing)} (saw {sorted(seen)})")

    if "tools_used_contains_any" in expect:
        # At least ONE of the listed tools ran — for questions with more
        # than one legitimate route (e.g. "my high priority open tasks"
        # is answerable via list_tasks OR get_my_focus_tasks; demanding
        # one exact tool made the case a chronic false FAIL while the
        # judge scored the answer 1.0 across the board).
        accepted = set(expect["tools_used_contains_any"])
        if not accepted & set(tools_used):
            _add(
                f"tools_used_contains_any: none of {sorted(accepted)} ran "
                f"(saw {sorted(set(tools_used))})"
            )

    if "tools_used_excludes" in expect:
        forbidden = set(expect["tools_used_excludes"])
        seen = set(tools_used)
        leaked = forbidden & seen
        if leaked:
            _add(f"tools_used_excludes: forbidden tool was used: {sorted(leaked)}")

    if "pending_tools_contains" in expect:
        required = set(expect["pending_tools_contains"])
        seen = set(tools_pending)
        missing = required - seen
        if missing:
            _add(f"pending_tools_contains: missing {sorted(missing)} (saw {sorted(seen)})")

    if "answer_contains_any" in expect:
        needles = [s.lower() for s in expect["answer_contains_any"]]
        haystack = answer.lower()
        if not any(n in haystack for n in needles):
            _add(f"answer_contains_any: none of {needles} found in answer")

    if "answer_does_not_contain" in expect:
        forbidden = [s.lower() for s in expect["answer_does_not_contain"]]
        haystack = answer.lower()
        matched = [s for s in forbidden if s in haystack]
        if matched:
            _add(f"answer_does_not_contain: matched {matched}")

    if "citations_contain" in expect:
        required = {c.lower() for c in expect["citations_contain"]}
        missing = required - citations_seen
        if missing:
            _add(
                f"citations_contain: missing {sorted(missing)} "
                f"(found {sorted(citations_seen)})"
            )

    if "has_citations" in expect and expect["has_citations"]:
        if not citations_seen:
            _add("has_citations: answer contains no [entity_id] citations")

    if "no_citations" in expect and expect["no_citations"]:
        if citations_seen:
            _add(f"no_citations: answer contains citations: {sorted(citations_seen)}")

    if "citations_count_at_least" in expect:
        n = int(expect["citations_count_at_least"])
        if len(citations_seen) < n:
            _add(
                f"citations_count_at_least: got {len(citations_seen)} citation(s), "
                f"expected >= {n} (saw {sorted(citations_seen)})"
            )

    if "answer_contains_all" in expect:
        needles = [s.lower() for s in expect["answer_contains_all"]]
        haystack = answer.lower()
        missing = [n for n in needles if n not in haystack]
        if missing:
            _add(f"answer_contains_all: missing {missing} from answer")

    if "answer_length_at_least" in expect:
        n = int(expect["answer_length_at_least"])
        if len(answer) < n:
            _add(
                f"answer_length_at_least: got {len(answer)} chars, expected >= {n} "
                f"(answer was: {answer!r})"
            )

    if "tool_call_errors_contain" in expect:
        substrs = [s.lower() for s in expect["tool_call_errors_contain"]]
        error_msgs = [(e.get("error") or "").lower() for e in tool_errors]
        for needle in substrs:
            if not any(needle in msg for msg in error_msgs):
                _add(
                    f"tool_call_errors_contain: no tool_call_error matched {needle!r} "
                    f"(errors: {error_msgs})"
                )

    if "tool_call_errors_contain_any" in expect:
        # ANY-of variant — for denials whose exact phrasing is a security
        # choice, not a contract: chat ACL errors deliberately say
        # "not found or has no members" so an outsider can't probe which
        # chat ids exist. The case asserts the request was refused, not
        # the refusal's wording.
        substrs = [s.lower() for s in expect["tool_call_errors_contain_any"]]
        error_msgs = [(e.get("error") or "").lower() for e in tool_errors]
        if not any(needle in msg for needle in substrs for msg in error_msgs):
            _add(
                f"tool_call_errors_contain_any: no tool_call_error matched any of "
                f"{substrs} (errors: {error_msgs})"
            )

    if "no_errors" in expect and expect["no_errors"]:
        if fatal_errors:
            msgs = [e.get("message") for e in fatal_errors]
            _add(f"no_errors: saw fatal error events: {msgs}")

    if "step_count_at_most" in expect:
        n = int(expect["step_count_at_most"])
        if step_count > n:
            _add(f"step_count_at_most: got {step_count}, expected <= {n}")

    return reasons


# --------------------------------------------------------------------------- #
# Retrieval mode (Phase 6 — direct search() assertions)                       #
# --------------------------------------------------------------------------- #


def _retrieval_metrics(entities: list[dict[str, Any]], expect: dict[str, Any]) -> dict[str, float]:
    """Continuous retrieval-quality signals on top of the binary pass/fail.

    Why MRR (rank) is the headline rather than recall@n: most gold sets in
    `retrieval_cases.yaml` are singletons (one entity / one title
    substring), so a recall@n fraction is just 0.0/1.0 — identical to the
    existing `must_contain_*` binary, trending nothing new. The *rank* at
    which the gold item lands distinguishes "surfaced at #1" from
    "surfaced at #5" — both pass the binary today but are very different
    retrieval outcomes, and a retrieval change that lifts gold from rank 4
    to rank 2 is invisible to pass/fail. `recall_at_n` is still reported
    but is only fractional for the handful of multi-gold cases.

    Scope (do not overclaim): retrieval cases run under `mode="eval"`
    with production overlays OFF (freshness, chunk-type weights, LLM
    reranker, graph expansion — see `run_retrieval_case`; per-case
    `overlays: true` opts back in), so by default these measure RAW
    BM25+vector+RRF recall on fixtures. They are ideal
    for A/B-ing a retrieval change, but are NOT production recall — that
    is the online-sampling half of the foundation, which this doesn't
    touch. Ranks are measured within the returned list (capped at the
    case's `limit`); gold outside it scores reciprocal rank 0.

    Tool-selection accuracy is intentionally NOT emitted here: every
    `tools_used_contains` in `cases.yaml` is a singleton, so a "tool
    recall" number would be purely binary (a fraction in a binary
    costume). A continuous tool-selection metric is blocked on authoring
    multi-tool / negative-tool gold cases first.

    Returns `{}` for cases with no rank-checkable gold (e.g. a pure
    `must_not_contain_title` adversarial case) so the caller skips them.
    """
    ranked_titles = [(e.get("title") or "").lower() for e in entities]
    ranked_ids = [e.get("entity_id") for e in entities]

    def _rank_of_title(needle: str) -> int | None:
        needle = (needle or "").lower()
        if not needle:
            return None
        for i, t in enumerate(ranked_titles):
            if needle in t:
                return i + 1  # 1-indexed
        return None

    def _rank_of_id(eid: Any) -> int | None:
        for i, x in enumerate(ranked_ids):
            if x == eid:
                return i + 1
        return None

    # Each gold "slot" contributes (reciprocal_rank, hit_within_declared_n).
    slots: list[tuple[float, bool]] = []

    needle = expect.get("top_result_title_contains")
    if isinstance(needle, str) and needle:
        r = _rank_of_title(needle)
        slots.append((1.0 / r if r else 0.0, bool(r and r <= 1)))

    spec = expect.get("must_contain_title_in_top_n")
    if isinstance(spec, dict):
        n = int(spec.get("n", 0))
        for s in spec.get("title_substrings") or []:
            r = _rank_of_title(s)
            slots.append((1.0 / r if r else 0.0, bool(r and r <= n)))

    spec = expect.get("must_contain_in_top_n")
    if isinstance(spec, dict):
        n = int(spec.get("n", 0))
        for eid in spec.get("entity_ids") or []:
            r = _rank_of_id(eid)
            slots.append((1.0 / r if r else 0.0, bool(r and r <= n)))

    # OR matcher: one slot, satisfied by the best-ranked candidate.
    spec = expect.get("must_contain_any_title_in_top_n")
    if isinstance(spec, dict):
        n = int(spec.get("n", 0))
        ranks = [r for r in (_rank_of_title(s) for s in spec.get("title_substrings") or []) if r]
        best = min(ranks) if ranks else None
        slots.append((1.0 / best if best else 0.0, bool(best and best <= n)))

    out: dict[str, float] = {}
    if slots:
        mrr = sum(rr for rr, _ in slots) / len(slots)
        recall = sum(1.0 for _, hit in slots if hit) / len(slots)
        out["mrr"] = round(mrr, 4)
        out["recall_at_n"] = round(recall, 4)

    out.update(_precision_at_k(ranked_titles, ranked_ids, expect))
    return out


def _precision_at_k(
    ranked_titles: list[str], ranked_ids: list[Any], expect: dict[str, Any]
) -> dict[str, float]:
    """Continuous retrieval-PRECISION signal (Q0.1 — §4.4: 'no continuous
    precision metric exists' / the unmeasured ~50%-chip-quality north-star).

    Recall asks "did the gold surface?"; precision asks "is the surfaced set
    free of noise?" — the opposite pull. With the singleton gold most cases
    declare, a naive precision@k is just 1/k whenever recall hits (a binary in
    a costume — the same trap `_retrieval_metrics` avoids for recall). So this
    is emitted ONLY when the case can actually move it: it declares >= 2
    RELEVANT markers (a cluster the top-k should be packed with) OR explicit
    NEGATIVES (`must_not_contain[_title]`, known distractors that must not
    crowd in). Otherwise returns `{}` and the metric is skipped for that case.

      * precision_at_k  = |relevant in top-k| / k
        Falls as junk/distractors fill the window even when recall stays 1.0.
      * labeled_precision = |relevant| / (|relevant| + |known-irrelevant|)
        over only the LABELED results in the top-k — emitted when the case
        declares negatives, so a distractor that sneaks in is penalised
        directly (independent of how many unlabeled neutrals are present).

    `k` is the case's `precision_k`, else the widest declared top-n window,
    else the returned count.
    """
    relevant_titles: list[str] = []
    relevant_ids: list[Any] = []
    ns: list[int] = []

    needle = expect.get("top_result_title_contains")
    if isinstance(needle, str) and needle:
        relevant_titles.append(needle.lower())

    for key in ("must_contain_title_in_top_n", "must_contain_any_title_in_top_n"):
        spec = expect.get(key)
        if isinstance(spec, dict):
            ns.append(int(spec.get("n", 0)))
            relevant_titles.extend((s or "").lower() for s in spec.get("title_substrings") or [])

    spec = expect.get("must_contain_in_top_n")
    if isinstance(spec, dict):
        ns.append(int(spec.get("n", 0)))
        relevant_ids.extend(spec.get("entity_ids") or [])

    irrelevant_titles = [(s or "").lower() for s in expect.get("must_not_contain_title") or []]
    irrelevant_ids = list(expect.get("must_not_contain") or [])

    n_relevant = len(relevant_titles) + len(relevant_ids)
    has_negatives = bool(irrelevant_titles or irrelevant_ids)
    if n_relevant < 2 and not has_negatives:
        return {}

    k = int(expect.get("precision_k") or (max(ns) if ns else 0) or len(ranked_titles))
    if k < 2:
        return {}
    k = min(k, len(ranked_titles)) or k

    rel = irr = 0
    for i in range(min(k, len(ranked_titles))):
        title = ranked_titles[i]
        eid = ranked_ids[i] if i < len(ranked_ids) else None
        is_rel = any(s in title for s in relevant_titles) or (eid in relevant_ids)
        is_irr = any(s in title for s in irrelevant_titles) or (eid in irrelevant_ids)
        if is_rel:
            rel += 1
        elif is_irr:
            irr += 1

    out: dict[str, float] = {"precision_at_k": round(rel / k, 4)}
    labeled = rel + irr
    if has_negatives and labeled:
        out["labeled_precision"] = round(rel / labeled, 4)
    return out


def run_retrieval_case(case: dict[str, Any]) -> CaseResult:
    """Execute one retrieval case by calling `search(...)` directly.

    No agent loop, no LLM calls. The case YAML specifies a query +
    optional filters and a set of gold-standard assertions about
    which entities should appear (and at what rank) in the result.
    """
    case = _resolve_fixture(case)
    case_id = case.get("id") or "(unnamed)"
    query = case.get("query") or ""
    team_id = case.get("team_id") or ""
    user_id = case.get("user_id") or ""
    expect = case.get("expect") or {}

    if not query or not team_id or not user_id:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=0,
            failure_reasons=["case is missing query/team_id/user_id"],
        )

    started = time.monotonic()
    try:
        # Honor the same RAG_USE_QUERY_REWRITE flag the agent path
        # uses, so `agent_eval_compare --b-overrides
        # '{"RAG_USE_QUERY_REWRITE": true}'` actually exercises
        # rewriting on the retrieval suite. Lazy import — `settings`
        # is set up once Django has loaded.
        from django.conf import settings as _settings  # noqa: PLC0415

        result = search(
            query=query,
            team_id=team_id,
            user_id=user_id,
            entity_types=case.get("entity_types"),
            date_from=case.get("date_from"),
            date_to=case.get("date_to"),
            limit=int(case.get("limit", 10)),
            use_vector=bool(case.get("use_vector", True)),
            rewrite=bool(_settings.SEARCH_ENGINE.get("RAG_USE_QUERY_REWRITE", False)),
            # `mode="eval"` disables freshness boost, chunk-type
            # reweighting, AND the production overlays (LLM reranker +
            # graph expansion), so the retrieval-quality numbers reflect
            # raw BM25 + vector + RRF. A case that exists to measure an
            # overlay (e.g. the graph_native_* cases) opts back in with
            # `overlays: true` in its YAML — the overlay still honors its
            # own RAG_* flag, so `agent_eval_compare --b-overrides` can
            # flag-flip it for a clean A/B.
            mode="eval",
            overlays=bool(case.get("overlays", False)),
        )
    except Exception as e:  # noqa: BLE001
        duration_ms = int((time.monotonic() - started) * 1000)
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=duration_ms,
            failure_reasons=[f"search() crashed: {e!r}"],
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    entities = result.get("results", []) or []
    reasons = _check_retrieval_expectations(entities, expect)

    return CaseResult(
        case_id=case_id,
        passed=not reasons,
        duration_ms=duration_ms,
        failure_reasons=reasons,
        metrics=_retrieval_metrics(entities, expect),
    )


def _check_retrieval_expectations(
    entities: list[dict[str, Any]], expect: dict[str, Any]
) -> list[str]:
    """Assertions for retrieval-quality cases.

    Operates on the entity list returned by `search(...)`, where each
    entity has at least `entity_type`, `entity_id`, and `title`. Rank
    is 1-indexed in the failure messages so they read naturally.
    """
    reasons: list[str] = []
    ranked_ids = [e.get("entity_id") for e in entities]
    ranked_types = [e.get("entity_type") for e in entities]
    ranked_titles = [(e.get("title") or "") for e in entities]

    def _add(reason: str) -> None:
        reasons.append(reason)

    if "must_contain_in_top_n" in expect:
        spec = expect["must_contain_in_top_n"] or {}
        n = int(spec.get("n", 0))
        required = [eid for eid in (spec.get("entity_ids") or [])]
        top = set(ranked_ids[:n])
        missing = [eid for eid in required if eid not in top]
        if missing:
            _add(
                f"must_contain_in_top_n: missing {missing} from top {n} "
                f"(top {n} was {ranked_ids[:n]})"
            )

    # Title-substring AND matcher — every entry in `title_substrings`
    # must appear (case-insensitive) in the title of some entity in
    # the top N. Use when ALL of several expected entities need to
    # surface together. Robust across reseedings.
    if "must_contain_title_in_top_n" in expect:
        spec = expect["must_contain_title_in_top_n"] or {}
        n = int(spec.get("n", 0))
        required = [s.lower() for s in (spec.get("title_substrings") or [])]
        top_titles = [t.lower() for t in ranked_titles[:n]]
        missing = [needle for needle in required if not any(needle in t for t in top_titles)]
        if missing:
            _add(
                f"must_contain_title_in_top_n: missing {missing} from top {n} "
                f"(top {n} titles were {ranked_titles[:n]})"
            )

    # Title-substring OR matcher — AT LEAST ONE entry in
    # `title_substrings` must appear in the top N. Use when the
    # question has multiple acceptable answers and any of them
    # surfacing is a pass.
    if "must_contain_any_title_in_top_n" in expect:
        spec = expect["must_contain_any_title_in_top_n"] or {}
        n = int(spec.get("n", 0))
        candidates = [s.lower() for s in (spec.get("title_substrings") or [])]
        top_titles = [t.lower() for t in ranked_titles[:n]]
        any_match = any(needle in t for needle in candidates for t in top_titles)
        if candidates and not any_match:
            _add(
                f"must_contain_any_title_in_top_n: none of {candidates} found in top {n} "
                f"(top {n} titles were {ranked_titles[:n]})"
            )

    # Title-substring NEGATIVE matcher — none of these substrings may
    # appear as a title in the result set. Use for adversarial /
    # ACL-leak cases ("the off-team document must NOT surface").
    if "must_not_contain_title" in expect:
        forbidden = [s.lower() for s in (expect["must_not_contain_title"] or [])]
        leaked = [
            needle for needle in forbidden if any(needle in t.lower() for t in ranked_titles)
        ]
        if leaked:
            _add(
                f"must_not_contain_title: forbidden title substring(s) "
                f"appeared: {leaked} (titles: {ranked_titles})"
            )

    # Top-result title matcher — the #1 ranked entity must have a
    # title that contains this substring (case-insensitive).
    if "top_result_title_contains" in expect:
        needle = str(expect["top_result_title_contains"]).lower()
        top_title = ranked_titles[0].lower() if ranked_titles else ""
        if needle not in top_title:
            _add(
                f"top_result_title_contains: top hit title {top_title!r} "
                f"does not contain {needle!r}"
            )

    if "must_contain_entity_type_in_top_n" in expect:
        spec = expect["must_contain_entity_type_in_top_n"] or {}
        n = int(spec.get("n", 0))
        wanted = set(spec.get("entity_types") or [])
        top_types = set(ranked_types[:n])
        if not (wanted & top_types):
            _add(
                f"must_contain_entity_type_in_top_n: none of {sorted(wanted)} in top {n} "
                f"(saw types {sorted(top_types)})"
            )

    if "must_not_contain" in expect:
        forbidden = set(expect["must_not_contain"] or [])
        leaked = forbidden & set(ranked_ids)
        if leaked:
            _add(f"must_not_contain: forbidden entities present in results: {sorted(leaked)}")

    if "top_result_entity_type" in expect:
        want = expect["top_result_entity_type"]
        got = ranked_types[0] if ranked_types else None
        if got != want:
            _add(f"top_result_entity_type: top hit is {got!r}, expected {want!r}")

    if "result_count_at_least" in expect:
        n = int(expect["result_count_at_least"])
        if len(entities) < n:
            _add(f"result_count_at_least: got {len(entities)} results, expected >= {n}")

    if "result_count_at_most" in expect:
        n = int(expect["result_count_at_most"])
        if len(entities) > n:
            _add(f"result_count_at_most: got {len(entities)} results, expected <= {n}")

    return reasons


# --------------------------------------------------------------------------- #
# Summary mode (Q0.3 — summarizer-fidelity)                                   #
# --------------------------------------------------------------------------- #
#
# The thread/note summary seeds the agent's system prompt for follow-up Q&A
# (§3.11 / §4.8), so a low-fidelity summary silently degrades every downstream
# answer. This suite measures the PRODUCTION summary path: it summarises a
# source document with the SAME system prompt the live thread/note summariser
# uses (routed by `kind`), then scores summary-vs-source with `judge_summary`
# plus a deterministic entity-overlap metric. The scores ride the existing
# `metrics` channel (prefixed `summary_`) so they print, aggregate, and
# persist through the same plumbing as the retrieval metrics.


def _generate_summary(kind: str, source: str) -> str:
    """Summarise `source` with the production system prompt for `kind`
    (`thread` or `note`) — exercises the real summariser, not a stand-in.

    The case's `source` must already be in the shape that kind's
    `_format_*_for_prompt` produces (thread = "Name: text" lines; note =
    title + body), since this calls the model with the production system
    prompt directly. See `summary_cases.yaml` for examples.
    """
    from origin.search_engine.llm import get_model_client  # noqa: PLC0415
    from origin.search_engine.llm.types import AgentMessage  # noqa: PLC0415

    if kind == "note":
        from origin.search_engine.agent.note_summary import (  # noqa: PLC0415
            _SUMMARY_SYSTEM_PROMPT as sys_prompt,
        )
    else:
        from origin.search_engine.agent.thread_summary import (  # noqa: PLC0415
            _SUMMARY_SYSTEM_PROMPT as sys_prompt,
        )

    client = get_model_client()
    chunks: list[str] = []
    for text, _fcall in client.generate_step(
        messages=[AgentMessage(role="user", text=source + "\n\nNow write the summary.")],
        tools=[],
        system_instruction=sys_prompt,
    ):
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def _check_summary_expectations(
    summary: str, judged: dict[str, Any], overlap: dict[str, float], expect: dict[str, Any]
) -> list[str]:
    """Optional gating assertions for a summary case. Most cases are pure
    measurement (no `expect` block → always passes); these let a case fail
    the CI gate on a hard floor or a prompt-injection containment check."""
    reasons: list[str] = []
    if "error" in judged:
        reasons.append(f"summary judge errored: {judged['error']}")
        return reasons

    hay = (summary or "").lower()

    if "min_fidelity" in expect:
        floor = float(expect["min_fidelity"])
        if judged.get("fidelity", 0.0) < floor:
            reasons.append(
                f"min_fidelity: fidelity {judged.get('fidelity', 0.0):.2f} < {floor:.2f}"
            )

    if "min_entity_recall" in expect and "entity_recall" in overlap:
        floor = float(expect["min_entity_recall"])
        if overlap["entity_recall"] < floor:
            reasons.append(
                f"min_entity_recall: entity_recall {overlap['entity_recall']:.2f} < {floor:.2f}"
            )

    if "must_contain_any" in expect:
        needles = [s.lower() for s in expect["must_contain_any"]]
        if needles and not any(n in hay for n in needles):
            reasons.append(f"must_contain_any: none of {needles} found in summary")

    if "must_not_contain" in expect:
        forbidden = [s.lower() for s in expect["must_not_contain"]]
        matched = [s for s in forbidden if s in hay]
        if matched:
            reasons.append(f"must_not_contain: summary contains {matched}")

    return reasons


def run_summary_case(case: dict[str, Any]) -> CaseResult:
    """Execute one summary-fidelity case. No fixture/DB needed — the source
    document is inline in the case — so this suite is fast and self-contained
    (one LLM call to summarise + one judge call)."""
    case_id = case.get("id") or "(unnamed)"
    kind = (case.get("kind") or "thread").lower()
    source = case.get("source") or ""
    expect = case.get("expect") or {}

    if not source.strip():
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=0,
            failure_reasons=["summary case is missing a non-empty `source`"],
        )

    # Lazy import — judge touches the LLM client; keep module import light.
    from origin.search_engine.agent.evals.judge import (  # noqa: PLC0415
        entity_overlap,
        judge_summary,
    )

    started = time.monotonic()
    try:
        summary = _generate_summary(kind, source)
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=int((time.monotonic() - started) * 1000),
            failure_reasons=[f"summary generation failed: {e!r}"],
        )

    if not summary:
        return CaseResult(
            case_id=case_id,
            passed=False,
            duration_ms=int((time.monotonic() - started) * 1000),
            failure_reasons=["summariser returned an empty summary"],
        )

    declared = expect.get("entities")
    judged = judge_summary(summary=summary, source=source, key_entities=declared)
    overlap = entity_overlap(summary, source, declared_entities=declared)
    duration_ms = int((time.monotonic() - started) * 1000)

    metrics: dict[str, float] = {}
    if "error" not in judged:
        metrics["summary_fidelity"] = round(float(judged.get("fidelity", 0.0)), 4)
        metrics["summary_coverage"] = round(float(judged.get("coverage", 0.0)), 4)
        metrics["summary_entity_preservation"] = round(
            float(judged.get("entity_preservation", 0.0)), 4
        )
    metrics.update(overlap)

    reasons = _check_summary_expectations(summary, judged, overlap, expect)

    return CaseResult(
        case_id=case_id,
        passed=not reasons,
        duration_ms=duration_ms,
        failure_reasons=reasons,
        query=f"[summarise:{kind}]",
        answer=summary,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Setup helpers                                                               #
# --------------------------------------------------------------------------- #


def _setup_inject_note(spec: dict[str, Any], *, team_id: str, user_id: str):
    """Index a transient adversarial note for the duration of a case.

    Used by prompt-injection cases. The note is pushed directly into
    the OpenSearch index (skipping the normal chunker pipeline) so the
    test is self-contained and doesn't require seeding the SQL DB. The
    returned callable removes the doc on teardown.

    `spec` keys:
        title (str): note title
        body  (str): attack payload (the body the model will see)
    """
    from origin.search_engine.embeddings import embed_one  # noqa: PLC0415
    from origin.search_engine.opensearch_client import (  # noqa: PLC0415
        get_client,
        get_index_alias,
    )

    title = (spec.get("title") or "Test injection note").strip()
    body = spec.get("body") or ""
    if not body:
        raise ValueError("inject_note.body is required")

    client = get_client()
    index = get_index_alias()
    chunk_id = f"eval-inject-note:{uuid.uuid4()}"
    note_id = -abs(hash(chunk_id)) % 10_000_000  # negative-ish to avoid real ids

    doc = {
        "chunk_id": chunk_id,
        "entity_type": "note",
        "entity_id": f"note:personal:{note_id}",
        "chunk_type": "note_title_body",
        "team_id": team_id,
        "user_id": user_id,
        "acl_user_ids": [user_id],
        "title": title,
        "snippet_text": body[:200],
        "search_text": f"{title}\n{body}",
        "embedding": embed_one(f"{title}\n{body}"),
        "note_id": str(note_id),
        "note_type": "personal",
        "index_schema_version": "v1",
    }
    client.index(index=index, id=chunk_id, body=doc, refresh="wait_for")

    def cleanup() -> None:
        try:
            client.delete(index=index, id=chunk_id, refresh="wait_for")
        except Exception:  # noqa: BLE001
            log.exception("Failed to delete eval-injected note %s", chunk_id)

    return cleanup


def _setup_seed_conversation(spec: dict[str, Any], *, team_id: str, user_id: str):
    """Index a transient PAST conversation into the per-user `conversation`
    lane for cross-session-memory cases (Q2.3). Pushed straight to OpenSearch
    (like `_setup_inject_note`) so the case is self-contained — it simulates a
    prior, ended AgentRun the `search_past_conversations` tool should recall.

    `spec` keys:
        question (str): the earlier question the user asked.
        answer   (str): the answer — should carry a DISTINCTIVE fact that is
                        NOT present anywhere in the indexed workspace, so the
                        only way to surface it is the conversation lane.
    """
    from origin.search_engine.embeddings import embed_one  # noqa: PLC0415
    from origin.search_engine.opensearch_client import (  # noqa: PLC0415
        get_client,
        get_index_alias,
    )

    question = (spec.get("question") or "").strip()
    answer = (spec.get("answer") or "").strip()
    if not answer:
        raise ValueError("seed_conversation.answer is required")

    client = get_client()
    index = get_index_alias()
    chunk_id = f"eval-conversation:{uuid.uuid4()}"
    search_text = f"Q: {question}\nA: {answer}"

    doc = {
        "chunk_id": chunk_id,
        "entity_type": "conversation",
        "entity_id": chunk_id,
        "chunk_type": "conversation",
        "team_id": team_id,
        "user_id": user_id,
        "acl_user_ids": [user_id],
        "title": question or "(past conversation)",
        "snippet_text": answer[:200],
        "search_text": search_text,
        "embedding": embed_one(search_text),
        "index_schema_version": "v1",
    }
    client.index(index=index, id=chunk_id, body=doc, refresh="wait_for")

    def cleanup() -> None:
        try:
            client.delete(index=index, id=chunk_id, refresh="wait_for")
        except Exception:  # noqa: BLE001
            log.exception("Failed to delete eval-seeded conversation %s", chunk_id)

    return cleanup
