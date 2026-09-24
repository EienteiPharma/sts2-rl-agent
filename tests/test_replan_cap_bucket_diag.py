"""Reporting-only replan_cap bucket diagnostics (Searcher split)."""

from __future__ import annotations

import numpy as np

from sts2_env.eval.combat_jev import CombatJevTelemetry, FAILOPEN_REPLAN_CAP
from sts2_env.eval.combat_turn_plan import (
    MAX_REPLANS_PER_PLAYER_TURN,
    REPLAN_CAP_BUCKET_SHORTLIST_IDLE,
    REPLAN_CAP_BUCKET_TRUE_EXHAUSTION,
    TurnPlanCandidate,
    TurnPlanTurnDiag,
    classify_replan_cap_bucket,
    record_turn_plan_replan_cap_bucket,
)
from sts2_env.eval.hold_turn_replay import HoldTurnPlanEpisodeReplay


def test_classify_true_replan_exhaustion_on_illegal_step_budget():
    diag = TurnPlanTurnDiag()
    for _ in range(MAX_REPLANS_PER_PLAYER_TURN + 1):
        diag.note_replan_trigger("illegal_step")
    diag.jev_plan_picks = 4
    diag.max_picked_plan_steps = 3
    assert classify_replan_cap_bucket(diag) == REPLAN_CAP_BUCKET_TRUE_EXHAUSTION


def test_classify_shortlist_idle_degenerate_legal():
    diag = TurnPlanTurnDiag()
    diag.degenerate_legal_key_picks = 2
    diag.jev_plan_picks = 2
    diag.max_picked_plan_steps = 1
    diag.single_step_only_shortlists = 2
    diag.note_replan_trigger("illegal_step")
    assert classify_replan_cap_bucket(diag) == REPLAN_CAP_BUCKET_SHORTLIST_IDLE


def test_telemetry_records_replan_cap_bucket():
    tel = CombatJevTelemetry()
    record_turn_plan_replan_cap_bucket(tel, REPLAN_CAP_BUCKET_TRUE_EXHAUSTION)
    report = tel.as_report()
    assert report["turn_plan_replan_cap_bucket"][REPLAN_CAP_BUCKET_TRUE_EXHAUSTION] == 1


def test_replay_turn_row_carries_replan_cap_bucket_from_shadow():
    rec = HoldTurnPlanEpisodeReplay.from_hold_job(
        {"seed": 1, "fixture_index": 0, "enc_id": 16, "bucket": "normal", "ep": 0}
    )
    rec.record_replan_cap_catastrophe(
        player_turn=2,
        board={"self": {"hp": 50, "max_hp": 80}},
        replan_count=4,
        shadow={
            "turn_plan_failopen": True,
            "turn_plan_failopen_reason": FAILOPEN_REPLAN_CAP,
            "replan_cap_bucket": REPLAN_CAP_BUCKET_TRUE_EXHAUSTION,
            "replan_cap_diag": {"replan_triggers": {"illegal_step": 4}},
        },
    )
    row = rec.turns[-1]
    assert row["replan_cap_hit"] is True
    assert row["replan_cap_bucket"] == REPLAN_CAP_BUCKET_TRUE_EXHAUSTION
    assert row["replan_cap_diag"]["replan_triggers"]["illegal_step"] == 4
