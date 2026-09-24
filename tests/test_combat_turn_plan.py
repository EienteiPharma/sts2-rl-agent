"""Combat turn-plan path A tip①: semantic keys, board, enumerated plans (offline)."""
from __future__ import annotations

import numpy as np
import pytest

import sts2_env.eval.combat_turn_plan as turn_plan
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.eval.combat_turn_plan import (
    CATA_FAILOPEN_ILLEGAL_PLAN,
    CHOICE_COMBAT_TURN_PLAN,
    MAX_PLAN_STEPS,
    SEMANTIC_END_TURN,
    TurnPlanCandidate,
    TurnPlanPromptConfig,
    MAX_TYPESAFE_CHOICE_PLANS,
    build_combat_turn_plan_choice_question,
    build_turn_plan_jev_state,
    cap_plans_for_typesafe_choice,
    choose_combat_turn_plan_action,
    enumerate_candidate_plans,
    gym_action_for_semantic_key,
    jev_turn_plan_questions,
    legal_semantic_keys,
    semantic_key_for_gym_action,
    serialize_combat_board_full,
)
from sts2_env.eval.jev_types import JevAnswer
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import encode_observation


class _PlanAdapter:
    def __init__(self, plan_id: str, *, conf: float | None = 0.9):
        self.plan_id = plan_id
        self.conf = conf

    def system_one(self, state, questions):
        return {
            CHOICE_COMBAT_TURN_PLAN: JevAnswer(
                status="ok", choice=self.plan_id, confidence=self.conf
            )
        }


class _Ppo:
    def predict(self, obs, action_masks=None, deterministic=True):
        valid = np.flatnonzero(np.asarray(action_masks) == 1)
        return int(valid[0]), None


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


def test_typesafe_choice_hard_cap_255_and_pruned_count():
    keys = tuple(f"k{i}" for i in range(24))
    plans = enumerate_candidate_plans(keys, max_steps=3, max_plans=512)
    assert len(plans) > MAX_TYPESAFE_CHOICE_PLANS
    capped, pruned = cap_plans_for_typesafe_choice(plans)
    assert len(capped) == MAX_TYPESAFE_CHOICE_PLANS
    assert pruned == len(plans) - MAX_TYPESAFE_CHOICE_PLANS
    q = build_combat_turn_plan_choice_question(capped)
    assert len(q["criteria"]) <= MAX_TYPESAFE_CHOICE_PLANS


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


def test_prompt_layers_default_off():
    cfg = TurnPlanPromptConfig()
    assert cfg.system_rules is False
    assert cfg.board_json is False
    assert cfg.human_exemplars is False
    state = build_turn_plan_jev_state({"self": {"hp": 1}}, prompt_config=cfg)
    assert "board" not in state
    assert state["mode"] == "combat_turn_plan"


def test_runner_executes_end_turn_plan_low_conf_ok():
    env, combat, mask = _reset()
    obs = encode_observation(combat)
    keys = legal_semantic_keys(combat, mask)
    end_only = next(
        p for p in enumerate_candidate_plans(keys, max_steps=1, max_plans=64) if p.steps == (SEMANTIC_END_TURN,)
    )
    local, shadow = choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=_PlanAdapter(end_only.plan_id, conf=0.05),
        combat_obs=obs,
        env=env,
    )
    env.close()
    assert local == ACTION_END_TURN
    assert shadow["turn_plan_failopen"] is False


def test_runner_illegal_plan_catastrophe_fail_open():
    env, combat, mask = _reset()
    obs = encode_observation(combat)
    ppo = _Ppo()
    local, shadow = choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        ppo,
        adapter=_PlanAdapter("plan_not_in_list"),
        combat_obs=obs,
        env=env,
    )
    env.close()
    assert int(mask[local]) == 1
    assert shadow["turn_plan_catastrophe"] is True
    assert shadow["turn_plan_failopen_reason"] == CATA_FAILOPEN_ILLEGAL_PLAN


def test_eval_combat_suite_accepts_jev_turn():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_combat_suite.py"
    spec = importlib.util.spec_from_file_location("eval_combat_suite", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--combat-policy", "jev-turn"])
    assert args.combat_policy == "jev-turn"
    assert mod.parse_args([]).combat_policy == "ppo"


def test_jev_turn_plan_telemetry_hook_and_report_rates():
    from sts2_env.eval.combat_jev import (
        COMBAT_JEV_TURN_CATASTROPHE_REASONS,
        CombatJevTelemetry,
        FAILOPEN_ILLEGAL_PLAN,
        FAILOPEN_TIMEOUT,
    )
    from sts2_env.eval.combat_turn_plan import record_jev_turn_plan_turn

    tel = CombatJevTelemetry()
    record_jev_turn_plan_turn(tel, fulfilled=True)
    record_jev_turn_plan_turn(tel, fulfilled=False, catastrophe_reason=FAILOPEN_TIMEOUT)
    record_jev_turn_plan_turn(tel, fulfilled=False, catastrophe_reason=FAILOPEN_ILLEGAL_PLAN)
    record_jev_turn_plan_turn(tel, fulfilled=False)  # low_conf guardrail shape

    report = tel.as_report(n_episodes=1)
    assert report["turn_plan_turns"] == 4
    assert report["turn_plan_fulfilled"] == 1
    assert report["turn_plan_catastrophe_failopen"] == 2
    assert report["jev_fulfilled_rate"] == 0.25
    assert report["catastrophe_failopen_rate"] == 0.5
    assert set(report["jev_turn_catastrophe_reason"]) == set(COMBAT_JEV_TURN_CATASTROPHE_REASONS)
    assert report["jev_turn_catastrophe_reason"][FAILOPEN_TIMEOUT] == 1
    assert report["jev_turn_catastrophe_reason"][FAILOPEN_ILLEGAL_PLAN] == 1

    record_jev_turn_plan_turn(None, fulfilled=True)  # no-op


def test_turn_plan_missing_hung_ppo_distinct_from_random_failopen():
    from sts2_env.eval.combat_jev import CombatJevTelemetry, FAILOPEN_MISSING_HUNG_PPO

    env, combat, mask = _reset()
    obs = encode_observation(combat)
    tel = CombatJevTelemetry()
    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        None,
        adapter=_PlanAdapter("plan_not_in_list"),
        combat_obs=obs,
        env=env,
        telemetry=tel,
    )
    env.close()
    report = tel.as_report()
    assert report["jev_turn_catastrophe_reason"][FAILOPEN_MISSING_HUNG_PPO] == 1


def test_turn_plan_adapter_error_records_sample_message():
    from sts2_env.eval.combat_jev import FAILOPEN_ERROR, CombatJevTelemetry

    class _BoomAdapter:
        def system_one(self, state, questions):
            raise RuntimeError("adapter boom for sentry")

    env, combat, mask = _reset()
    obs = encode_observation(combat)
    tel = CombatJevTelemetry()
    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=_BoomAdapter(),
        combat_obs=obs,
        env=env,
        telemetry=tel,
    )
    env.close()
    report = tel.as_report()
    assert report["jev_turn_catastrophe_reason"][FAILOPEN_ERROR] == 1
    assert report["jev_turn_catastrophe_error_samples"]
    assert "RuntimeError: adapter boom for sentry" in report["jev_turn_catastrophe_error_samples"][0]


def test_runner_illegal_plan_increments_catastrophe_telemetry():
    from sts2_env.eval.combat_jev import CombatJevTelemetry, FAILOPEN_ILLEGAL_PLAN

    env, combat, mask = _reset()
    obs = encode_observation(combat)
    tel = CombatJevTelemetry()
    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=_PlanAdapter("plan_not_in_list"),
        combat_obs=obs,
        env=env,
        telemetry=tel,
    )
    env.close()
    report = tel.as_report()
    assert report["turn_plan_turns"] == 1
    assert report["catastrophe_failopen_rate"] == 1.0
    assert report["jev_turn_catastrophe_reason"][FAILOPEN_ILLEGAL_PLAN] == 1
