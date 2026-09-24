"""hang | A | B HOLD contrast table + hang-gap STOP (reporting only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sts2_env.eval.bh_assist_train import (
    ASSIST_EVAL_MIN_BOSS_PP,
    ASSIST_EVAL_MIN_OVERALL_PP,
)

HANG_GAP_STOP_PP = 3.0
_BUCKET_KEYS = ("overall", "elite", "boss")


def _rate(summary: Mapping[str, Any], bucket: str) -> float:
    block = summary.get(bucket) or {}
    return float(block.get("win_rate") or 0.0)


def pp_delta(new_rate: float, base_rate: float) -> float:
    return round((float(new_rate) - float(base_rate)) * 100.0, 2)


def load_hold_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"summary JSON must be an object: {path}")
    return data


def contrast_hang_a_b(
    hang: Mapping[str, Any],
    arm_a: Mapping[str, Any],
    arm_b: Mapping[str, Any],
    *,
    hang_gap_stop_pp: float = HANG_GAP_STOP_PP,
) -> dict[str, Any]:
    """Three-arm table: hang (ppo/bh_v1), A (jev-turn assist off), B (assist on)."""
    table: dict[str, dict[str, float]] = {}
    delta_a_hang: dict[str, float] = {}
    delta_b_hang: dict[str, float] = {}
    delta_assist: dict[str, float] = {}
    for key in _BUCKET_KEYS:
        h = _rate(hang, key)
        a = _rate(arm_a, key)
        b = _rate(arm_b, key)
        table[key] = {"hang": round(h, 4), "a": round(a, 4), "b": round(b, 4)}
        delta_a_hang[key] = pp_delta(a, h)
        delta_b_hang[key] = pp_delta(b, h)
        delta_assist[key] = pp_delta(b, a)

    overall_a = delta_a_hang["overall"]
    overall_b = delta_b_hang["overall"]
    stop = overall_a <= -hang_gap_stop_pp or overall_b <= -hang_gap_stop_pp
    stop_reasons: list[str] = []
    if overall_a <= -hang_gap_stop_pp:
        stop_reasons.append(
            f"overall_delta_a_vs_hang_pp={overall_a} <= -{hang_gap_stop_pp}"
        )
    if overall_b <= -hang_gap_stop_pp:
        stop_reasons.append(
            f"overall_delta_b_vs_hang_pp={overall_b} <= -{hang_gap_stop_pp}"
        )

    assist_overall_pp = delta_assist["overall"]
    assist_boss_pp = delta_assist["boss"]
    assist_effective = (
        assist_overall_pp >= float(ASSIST_EVAL_MIN_OVERALL_PP)
        and assist_boss_pp >= float(ASSIST_EVAL_MIN_BOSS_PP)
    )

    return {
        "columns": ("hang", "a", "b"),
        "win_rate_table": table,
        "delta_a_vs_hang_pp": delta_a_hang,
        "delta_b_vs_hang_pp": delta_b_hang,
        "delta_assist_b_minus_a_pp": delta_assist,
        "hang_gap_stop": {
            "triggered": bool(stop),
            "threshold_pp": float(hang_gap_stop_pp),
            "reasons": stop_reasons,
        },
        "assist_effectiveness_bar": {
            "min_overall_pp": ASSIST_EVAL_MIN_OVERALL_PP,
            "min_boss_pp": ASSIST_EVAL_MIN_BOSS_PP,
            "overall_pp": assist_overall_pp,
            "boss_pp": assist_boss_pp,
            "passed": bool(assist_effective),
        },
    }


def format_contrast_table(result: Mapping[str, Any]) -> str:
    table = result.get("win_rate_table") or {}
    lines = ["bucket | hang | A | B"]
    for key in _BUCKET_KEYS:
        row = table.get(key) or {}
        lines.append(
            f"{key} | {100.0 * float(row.get('hang', 0)):.1f}% | "
            f"{100.0 * float(row.get('a', 0)):.1f}% | "
            f"{100.0 * float(row.get('b', 0)):.1f}%"
        )
    da = result.get("delta_a_vs_hang_pp") or {}
    db = result.get("delta_b_vs_hang_pp") or {}
    d_ab = result.get("delta_assist_b_minus_a_pp") or {}
    lines.append(
        f"Δ vs hang (pp) A: overall {da.get('overall')} boss {da.get('boss')}"
    )
    lines.append(
        f"Δ vs hang (pp) B: overall {db.get('overall')} boss {db.get('boss')}"
    )
    lines.append(
        f"Δ assist B−A (pp): overall {d_ab.get('overall')} boss {d_ab.get('boss')}"
    )
    stop = result.get("hang_gap_stop") or {}
    lines.append(f"STOP hang-gap: {stop.get('triggered')} {stop.get('reasons')}")
    bar = result.get("assist_effectiveness_bar") or {}
    lines.append(f"Assist +3/+2 bar passed: {bar.get('passed')}")
    return "\n".join(lines)


__all__ = [
    "HANG_GAP_STOP_PP",
    "contrast_hang_a_b",
    "format_contrast_table",
    "load_hold_summary",
    "pp_delta",
]
