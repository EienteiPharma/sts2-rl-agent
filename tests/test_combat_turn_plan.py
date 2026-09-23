"""Combat turn-plan path A tip①: semantic keys, board, enumerated plans (offline)."""
from __future__ import annotations

import numpy as np
import pytest

import sts2_env.eval.combat_turn_plan as turn_plan
from sts2_env.eval.combat_turn_plan import (
    CHOICE_COMBAT_TURN_PLAN,
    MAX_PLAN_STEPS,
    SEMANTIC_END_TURN,
    TurnPlanCandidate,
    build_combat_turn_plan_choice_question,
    enumerate_candidate_plans,
    gym_action_for_semantic_key,
    jev_turn_plan_questions,
    legal_semantic_keys,
    semantic_key_for_gym_action,
    serialize_combat_board_full,
)
from sts2_env.gym_env.combat_env import STS2CombatEnv


def test_api_is_turn_plan_not_stepwise_combat_step():
    assert CHOICE_COMBAT_TURN_PLAN == "combat_turn_plan_choice"
    assert not hasattr(turn_plan, "choose_combat_step")
    assert not hasattr(turn_plan, "combat_action_id")
    assert "combat_step_choice" not in turn_plan.COMBAT_TURN_PLAN_INSTRUCTIONS


def test_toy_enumerator_respects_max_len_end_turn_and_legal_keys():
    keys = ("end_turn", "play:A:h0@self", "play:B:h1@self")
    plans = enumerate_candidate_plans(keys, max_steps=8, max_plans=200)
    assert plans
    assert all(1 <= len(p.steps) <= 8 for p in plans)
    assert all(all(s in keys for s in p.steps) for p in plans)
    for p in plans:
        if SEMANTIC_END_TURN in p.steps:
            assert p.steps[-1] == SEMANTIC_END_TURN
    ids = [p.plan_id for p in plans]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_enumerator_deterministic_and_bounded():
    keys = tuple(f"k{i}" for i in range(5))
    a = enumerate_candidate_plans(keys, max_steps=3, max_plans=10)
    b = enumerate_candidate_plans(keys, max_steps=3, max_plans=10)
    assert a == b
    assert len(a) == 10


def test_choice_question_uses_plan_ids_only():
    plans = (
        TurnPlanCandidate("plan_0000", ("play:X:h0@self",)),
        TurnPlanCandidate("plan_0001", (SEMANTIC_END_TURN,)),
    )
    board = {"self": {"hp": 50}}
    q = build_combat_turn_plan_choice_question(plans)
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"plan_0000", "plan_0001"}
    assert "plan_id" in q["instructions"]
    wrapped = jev_turn_plan_questions(board, plans)
    assert CHOICE_COMBAT_TURN_PLAN in wrapped


def _reset(seed: int = 3):
    env = STS2CombatEnv()
    obs, info = env.reset(seed=seed)
    combat = env.combat
    assert combat is not None
    mask = np.asarray(info["action_mask"])
    return env, combat, mask


def test_semantic_mapping_round_trip_on_legal_mask():
    env, combat, mask = _reset()
    keys = legal_semantic_keys(combat, mask)
    assert SEMANTIC_END_TURN in keys
    for key in keys:
        idx = gym_action_for_semantic_key(combat, mask, key)
        assert idx is not None
        assert int(mask[idx]) == 1
        assert semantic_key_for_gym_action(combat, idx) == key
    bad = gym_action_for_semantic_key(combat, mask, "play:NOT_A_REAL_CARD:h99@e0")
    assert bad is None
    env.close()


def test_serializer_includes_full_hand_piles_intents_no_truncation_keys():
    env, combat, mask = _reset()
    board = serialize_combat_board_full(combat, mask)
    env.close()
    assert "truncated" not in board
    assert "hand" in board["self"]
    assert len(board["self"]["hand"]) == len(combat.hand)
    assert "piles" in board
    for pile in ("draw", "discard", "exhaust"):
        assert pile in board["piles"]
        assert f"{pile}_n" in board["piles"]
    assert board["turn"]["end_turn_legal"] is True
    if board["enemies"]:
        assert "intent" in board["enemies"][0]
        assert "vuln" in board["enemies"][0]


def test_plans_from_live_legal_keys_max_steps():
    env, combat, mask = _reset()
    keys = legal_semantic_keys(combat, mask)
    plans = enumerate_candidate_plans(keys, max_steps=MAX_PLAN_STEPS, max_plans=64)
    env.close()
    assert plans
    assert plans[0].plan_id == "plan_0000"
    assert len(plans[0].steps) >= 1
