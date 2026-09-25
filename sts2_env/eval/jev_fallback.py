"""Client-layer confidence, score assist, and fallback threshold helpers.

Evaluates choice confidence thresholds, local HP pressure fallback formula,
and rest vs continue overrides. (MAP low-HP routing lives in map_lowhp.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.eval.jev_types import (
    CHOICE_CONFIDENCE_MIN,
    HP_PRESSURE_CONTINUE,
    HP_PRESSURE_REST,
)

if TYPE_CHECKING:
    from sts2_env.eval.jev_types import JevAnswer


def local_hp_pressure(hp: int, max_hp: int) -> float:
    """Fallback Score-shaped pressure: 1.0 at full HP, 2.0 at half HP."""
    return float(max_hp) / float(max(int(hp), 1))


def apply_choice_confidence(
    answer: JevAnswer,
    min_conf: float | None = None,
) -> JevAnswer:
    """Validate Choice answer against confidence threshold; mark uncertain on soft confidence."""
    if answer.status != "ok":
        return answer
    threshold = CHOICE_CONFIDENCE_MIN if min_conf is None else min_conf
    conf = answer.confidence
    if conf is None or conf < threshold:
        answer.status = "uncertain"
        answer.fallback_reason = f"choice confidence {conf} < {threshold}"
    return answer


def rest_or_continue_override(hp_pressure: float) -> str | None:
    """Return 'rest', 'continue', or None (trust Choice)."""
    if hp_pressure >= HP_PRESSURE_REST:
        return "rest"
    if hp_pressure <= HP_PRESSURE_CONTINUE:
        return "continue"
    return None


__all__ = [
    "apply_choice_confidence",
    "local_hp_pressure",
    "rest_or_continue_override",
]
