"""Jev-turn TypeSafe payload short context (Lab)."""

from __future__ import annotations

import json

from sts2_env.eval.combat_turn_plan import (
    RICH_TURN_PLAN_PROMPT_CONFIG,
    TURN_PLAN_JEV_PAYLOAD_MAX_BYTES,
    TurnPlanCandidate,
    build_turn_plan_typesafe_request,
    cap_plans_for_turn_plan_choice,
    enumerate_candidate_plans,
    jev_turn_plan_questions,
    turn_plan_typesafe_payload_bytes,
)
from tests.test_combat_turn_plan import _board_lethal_attack


def _fixture_plans(n_cap: int = 32):
    keys = tuple(
        f"k{i}" for i in range(10)
    )  # synthetic keys — size stress without combat env
    raw = enumerate_candidate_plans(keys, max_steps=3, max_plans=128)
    board = _board_lethal_attack()
    top, _, _, _ = cap_plans_for_turn_plan_choice(raw, board, max_choices=n_cap)
    return board, top


def test_short_payload_smaller_than_rich_and_under_cap():
    board, plans = _fixture_plans(32)
    short_state, short_q = build_turn_plan_typesafe_request(board, plans)
    rich_state, rich_q = build_turn_plan_typesafe_request(
        board, plans, prompt_config=RICH_TURN_PLAN_PROMPT_CONFIG
    )
    short_b = turn_plan_typesafe_payload_bytes(short_state, short_q)
    rich_b = turn_plan_typesafe_payload_bytes(rich_state, rich_q)
    assert short_b < rich_b
    assert short_b <= TURN_PLAN_JEV_PAYLOAD_MAX_BYTES
    assert "board" not in short_state
    assert "decision" in short_state
    assert set(short_state["decision"]) <= {
        "hp",
        "max_hp",
        "block",
        "energy",
        "incoming",
        "end_turn_ok",
    }
    crit = short_q["combat_turn_plan_choice"]["criteria"]
    assert all(len(v) <= 48 for v in crit.values())
    assert all("Play " not in v for v in crit.values())


def test_clamp_enforces_max_bytes_on_artificially_long_criteria():
    board = _board_lethal_attack()
    plans = tuple(
        TurnPlanCandidate(
            f"plan_{i:04d}",
            ("play:VERYLONGCARDNAME_IRONCLAD:h0@e0", "end_turn"),
        )
        for i in range(40)
    )
    _, questions = build_turn_plan_typesafe_request(board, plans)
    body = json.dumps(
        {"state": {"mode": "x"}, "questions": questions},
        separators=(",", ":"),
    )
    assert len(body.encode("utf-8")) <= TURN_PLAN_JEV_PAYLOAD_MAX_BYTES + 256
