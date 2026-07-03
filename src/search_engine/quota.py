"""Per-tier daily quota helpers.

A single `ModelUsageCounter` table holds counts for THREE quota
dimensions, distinguished by the `model_name` field which acts as a
polymorphic key:

  - `"gemini-2.5-flash"`, `"claude-sonnet-4-6"`, ...
        → per-model agent ask count.
  - `LLM_ASK_KEY = "__llm_ask__"`
        → daily total of agent asks (any model).
  - `WEB_SEARCH_KEY = "__web_search__"`
        → daily total of Tavily web searches.

Tier resolution reads `CustomUser.tier` directly (free | pro | max).
Per-tier limits live in `settings.SEARCH_ENGINE["TIER_QUOTAS"]`.

Race note: `(check_remaining, increment_usage)` is not atomic. Two
concurrent asks at 9/10 both pass the pre-check and both increment,
yielding 11/10. Accepted for v1 — the over-count is at most the
worker's concurrent-request count and the next call still gets
blocked. If this matters, wrap the pair in `select_for_update` inside
`transaction.atomic`.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.db.models import F
from django.utils import timezone

from origin.models.common.usage_models import ModelUsageCounter
from origin.models.common.user_models import CustomUser

log = logging.getLogger(__name__)

# Sentinel keys for the cross-dimensional counters in `ModelUsageCounter`.
# Chosen with leading + trailing underscores so they can never collide
# with a real model id from `MODEL_CATALOG`.
LLM_ASK_KEY = "__llm_ask__"
WEB_SEARCH_KEY = "__web_search__"


def get_user_tier(user_id: str) -> str:
    """Return the user's tier ('free' | 'pro' | 'max').

    Falls back to 'free' if the user can't be loaded — defensive
    against bad input; never raises.
    """
    try:
        tier = CustomUser.objects.filter(id=user_id).values_list("tier", flat=True).first()
    except Exception:  # noqa: BLE001
        log.exception("get_user_tier failed for user_id=%s", user_id)
        return "free"
    return tier or "free"


def _tier_cfg(tier: str) -> dict:
    """Return the TIER_QUOTAS dict for `tier`, or `free`'s if missing."""
    all_tiers = settings.SEARCH_ENGINE.get("TIER_QUOTAS") or {}
    return all_tiers.get(tier) or all_tiers.get("free") or {}


def get_quota(user_id: str, key: str) -> int | None:
    """Return the daily quota for this user + counter key.

    Returns `None` to mean "no quota applies" (treated as unlimited at
    enforcement sites).

    Dispatch:
      - `LLM_ASK_KEY`    → `tier_cfg["llm_ask_daily"]`.
      - `WEB_SEARCH_KEY` → `tier_cfg["web_search_daily"]`.
      - any model id    → `tier_cfg["model_daily"].get(key)`.
    """
    cfg = _tier_cfg(get_user_tier(user_id))
    if key == LLM_ASK_KEY:
        v = cfg.get("llm_ask_daily")
    elif key == WEB_SEARCH_KEY:
        v = cfg.get("web_search_daily")
    else:
        v = (cfg.get("model_daily") or {}).get(key)
    if v is None:
        return None
    return int(v)


def get_used_today(user_id: str, key: str) -> int:
    """Today's (UTC) count for this user + key. 0 if no row yet."""
    today = timezone.now().date()
    row = (
        ModelUsageCounter.objects.filter(user_id=user_id, model_name=key, usage_date=today)
        .only("count")
        .first()
    )
    return int(row.count) if row else 0


def check_remaining(user_id: str, key: str) -> tuple[bool, int, int | None]:
    """Return (allowed, used_today, limit_or_None).

    - `allowed=True` when no quota applies (limit is None) or
      `used_today < limit`.
    - `allowed=False` when the quota is exhausted.
    """
    limit = get_quota(user_id, key)
    used = get_used_today(user_id, key)
    if limit is None:
        return True, used, None
    return used < limit, used, limit


def increment_usage(user_id: str, key: str) -> None:
    """Atomically increment today's counter for (user, key).

    Failures are swallowed and logged — a counter write must never
    block the user's actual request.
    """
    today = timezone.now().date()
    try:
        obj, created = ModelUsageCounter.objects.get_or_create(
            user_id=user_id,
            model_name=key,
            usage_date=today,
            defaults={"count": 1},
        )
        if not created:
            ModelUsageCounter.objects.filter(pk=obj.pk).update(count=F("count") + 1)
    except Exception:  # noqa: BLE001
        log.exception("increment_usage failed for user=%s key=%s", user_id, key)
