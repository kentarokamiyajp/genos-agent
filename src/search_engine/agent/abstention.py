"""Abstention primitives (SPOTLIGHT_QUALITY_ARCHITECTURE.md §4.1).

Single source of truth for the "insufficient evidence → say so" path,
shared by the controller's programmatic abstention gate and the eval
runner's abstention-correctness metric. Kept dependency-free (stdlib
only) so controller → abstention and runner → abstention can't form an
import cycle.

`is_abstention` is intentionally NARROW: it matches absence-of-answer
phrasing ("couldn't find", "no information"), NOT generic negation. In
particular "you have no overdue tasks" is a grounded ANSWER, not an
abstention — it must NOT match, or the metric would score correct
zero-result answers as abstentions and the gate logic that reuses this
detector would misjudge them.
"""

from __future__ import annotations

# The canned reply the gate substitutes when the agent answered without
# any grounding. Contains the literal "couldn't find" so it both passes
# `is_abstention` below and keeps the existing `no_result_*` eval
# assertions (which look for that phrasing) green when the gate is on.
ABSTAIN_MESSAGE = (
    "I couldn't find anything in your workspace that answers that. "
    "It may help to rephrase, or to check that the relevant chats, "
    "tasks, or notes exist."
)

# Absence-of-answer markers. Lowercased substring match. Deliberately
# excludes bare "no"/"none" so grounded zero-count answers ("no overdue
# tasks", "none are blocked") are not misread as abstentions.
_ABSTAIN_PHRASES = (
    "couldn't find",
    "could not find",
    "couldn't locate",
    "could not locate",
    "unable to find",
    "wasn't able to find",
    "was not able to find",
    "no information",
    "no relevant",
    "no matching",
    "no results",
    "no record",
    "not in your workspace",
    "nothing in your workspace",
    "don't have any information",
    "do not have any information",
    "couldn't find anything",
)


def is_abstention(text: str | None) -> bool:
    """True when `text` declines to answer for lack of evidence.

    Narrow by design — see module docstring. Used by the abstention
    metric (did the agent abstain when it should have?) and by the gate
    (so a model that already abstained well isn't overwritten with the
    canned message).
    """
    t = (text or "").lower()
    return any(p in t for p in _ABSTAIN_PHRASES)
