"""Telemetry and shadow logging helpers for Jev non-combat evaluations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sts2_env.eval.map_lowhp import (
    MAP_LOWHP_HARD_REASON,
    MAP_LOWHP_SOFT_B_REASON,
)
from sts2_env.eval.jev_types import (
    EVENT_SAFE_FALLBACK_REASON,
    POTION_OR_RELIC_SAFE_REASON,
)

if TYPE_CHECKING:
    from sts2_env.eval.jev_types import JevAnswer


def format_shadow_log(answer: JevAnswer) -> dict[str, Any]:
    """Convert a JevAnswer into a standardized shadow log record dict."""
    log: dict[str, Any] = {
        "shadow_suggestion": answer.choice,
        "shadow_status": answer.status,
        "shadow_confidence": answer.confidence,
        "shadow_hp_pressure": answer.score,
        "shadow_fallback_reason": answer.fallback_reason,
    }
    if answer.card_fit is not None:
        log["jev_card_fit"] = answer.card_fit
    log["map_lowhp_hard"] = answer.fallback_reason == MAP_LOWHP_HARD_REASON
    log["map_lowhp_soft_b"] = answer.fallback_reason == MAP_LOWHP_SOFT_B_REASON
    log["event_safe_fallback"] = answer.fallback_reason == EVENT_SAFE_FALLBACK_REASON
    log["potion_or_relic_safe_fallback"] = answer.fallback_reason == POTION_OR_RELIC_SAFE_REASON
    return log


def default_shadow_fields(phase: str, *, jev_enabled: bool = False) -> dict[str, Any]:
    """Default stub/skipped shadow record fields for unexecuted or skipped phases."""
    return {
        "shadow_suggestion": None,
        "shadow_status": "ok" if jev_enabled else "stub",
        "shadow_confidence": None,
        "shadow_hp_pressure": None,
        "shadow_fallback_reason": None,
    }


__all__ = [
    "default_shadow_fields",
    "format_shadow_log",
]
