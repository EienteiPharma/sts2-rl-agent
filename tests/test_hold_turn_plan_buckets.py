"""Turn-plan episode bucket reporting."""
from __future__ import annotations

from sts2_env.eval.hold_turn_plan_buckets import (
    attach_turn_plan_episode_row_fields,
    summarize_turn_plan_episode_buckets,
)


def test_summarize_clean_vs_catastrophe_wr():
    rows = [
        {"win": True, "bucket": "elite", "had_turn_plan_catastrophe": False},
        {"win": False, "bucket": "boss", "had_turn_plan_catastrophe": True},
        {"win": True, "bucket": "boss", "had_turn_plan_catastrophe": False},
    ]
    out = summarize_turn_plan_episode_buckets(rows)
    assert out["episodes_clean"]["n"] == 2
    assert out["episodes_with_catastrophe"]["n"] == 1
    assert out["win_rate_clean"]["win_rate"] == 1.0
    assert out["win_rate_had_catastrophe"]["win_rate"] == 0.0
    assert out["boss_win_rate_clean"]["n"] == 1


def test_attach_row_fields():
    row: dict = {"win": False}
    attach_turn_plan_episode_row_fields(
        row,
        turn_plan_fields={
            "turn_plan_turns": 3,
            "turn_plan_fulfilled": 2,
            "turn_plan_catastrophe_failopen": 1,
        },
        catastrophe_reasons={"illegal_plan": 1},
    )
    assert row["had_turn_plan_catastrophe"] is True
    assert row["turn_plan_fulfilled_turns"] == 2
    assert row["turn_plan_catastrophe_reasons"] == {"illegal_plan": 1}
