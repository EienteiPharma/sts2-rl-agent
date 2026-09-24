"""Reporting-only HOLD buckets: catastrophe vs clean episodes (jev-turn)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _win_rate_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "win_rate": 0.0}
    wins = sum(1 for r in rows if r.get("win"))
    return {"n": n, "win_rate": round(wins / n, 4)}


def episode_had_turn_plan_catastrophe(row: Mapping[str, Any]) -> bool:
    if "had_turn_plan_catastrophe" in row:
        return bool(row.get("had_turn_plan_catastrophe"))
    return int(row.get("turn_plan_catastrophe_failopen") or 0) > 0


def attach_turn_plan_episode_row_fields(
    row: dict[str, Any],
    *,
    turn_plan_fields: Mapping[str, int],
    catastrophe_reasons: Mapping[str, int],
) -> None:
    cat_turns = int(turn_plan_fields.get("turn_plan_catastrophe_failopen") or 0)
    row["turn_plan_turns"] = int(turn_plan_fields.get("turn_plan_turns") or 0)
    row["turn_plan_fulfilled_turns"] = int(turn_plan_fields.get("turn_plan_fulfilled") or 0)
    row["turn_plan_catastrophe_turns"] = cat_turns
    row["had_turn_plan_catastrophe"] = cat_turns > 0
    row["turn_plan_catastrophe_reasons"] = {
        str(k): int(v) for k, v in catastrophe_reasons.items() if int(v) > 0
    }


def summarize_turn_plan_episode_buckets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split WR by whether the episode had any turn-plan catastrophe fail-open."""
    clean = [r for r in rows if not episode_had_turn_plan_catastrophe(r)]
    cat = [r for r in rows if episode_had_turn_plan_catastrophe(r)]
    boss_clean = [r for r in clean if str(r.get("bucket")) == "boss"]
    boss_cat = [r for r in cat if str(r.get("bucket")) == "boss"]

    turn_totals = {
        "turn_plan_turns": sum(int(r.get("turn_plan_turns") or 0) for r in rows),
        "turn_plan_fulfilled_turns": sum(
            int(r.get("turn_plan_fulfilled_turns") or 0) for r in rows
        ),
        "turn_plan_catastrophe_turns": sum(
            int(r.get("turn_plan_catastrophe_turns") or 0) for r in rows
        ),
    }

    return {
        "wr_any_note": "same as summary.overall win_rate (unchanged gate input)",
        "episodes_clean": {"n": len(clean)},
        "episodes_with_catastrophe": {"n": len(cat)},
        "win_rate_clean": _win_rate_subset(clean),
        "win_rate_had_catastrophe": _win_rate_subset(cat),
        "boss_win_rate_clean": _win_rate_subset(boss_clean),
        "boss_win_rate_had_catastrophe": _win_rate_subset(boss_cat),
        "episode_turn_totals": turn_totals,
    }


__all__ = [
    "attach_turn_plan_episode_row_fields",
    "episode_had_turn_plan_catastrophe",
    "summarize_turn_plan_episode_buckets",
]
