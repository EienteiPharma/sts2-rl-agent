"""Turn-plan execute path: replan trigger telemetry and hand-index remap."""

from __future__ import annotations

import numpy as np

from sts2_env.eval.combat_jev import CombatJevTelemetry
from sts2_env.eval.combat_turn_plan import (
    CHOICE_COMBAT_TURN_PLAN,
    TurnPlanCandidate,
    cap_plans_for_turn_plan_choice,
    choose_combat_turn_plan_action,
    enumerate_candidate_plans,
    legal_semantic_keys,
    player_turn_id,
    remap_plan_step_semantic_key,
    resolve_plan_step_for_execute,
    resolve_turn_plan_execute_remap,
    runtime_for_env,
    serialize_combat_board_full,
    SEMANTIC_END_TURN,
    TURN_PLAN_EXECUTE_REMAP_ENV,
)
from sts2_env.eval.jev_types import JevAnswer
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import encode_observation


def test_remap_plan_step_when_only_hand_index_drifted():
    legal = {
        "end_turn",
        "play:STRIKE_IRONCLAD:h0@e0",
        "play:DEFEND_IRONCLAD:h1@self",
    }
    stale = "play:STRIKE_IRONCLAD:h9@e0"
    assert stale not in legal
    assert remap_plan_step_semantic_key(stale, legal) == "play:STRIKE_IRONCLAD:h0@e0"


def test_remap_duplicate_card_picks_lowest_hand_index():
    legal = {
        "play:STRIKE_IRONCLAD:h0@e0",
        "play:STRIKE_IRONCLAD:h1@e0",
    }
    stale = "play:STRIKE_IRONCLAD:h9@e0"
    assert remap_plan_step_semantic_key(stale, legal) == "play:STRIKE_IRONCLAD:h0@e0"


def test_execute_remap_default_off():
    assert resolve_turn_plan_execute_remap() is False
    assert resolve_turn_plan_execute_remap(environ={TURN_PLAN_EXECUTE_REMAP_ENV: "0"}) is False
    assert resolve_turn_plan_execute_remap(environ={TURN_PLAN_EXECUTE_REMAP_ENV: "on"}) is True
    legal = {"play:STRIKE_IRONCLAD:h0@e0"}
    stale = "play:STRIKE_IRONCLAD:h9@e0"
    assert resolve_plan_step_for_execute(stale, legal, execute_remap=False) is None
    assert resolve_plan_step_for_execute(stale, legal, execute_remap=True) == (
        "play:STRIKE_IRONCLAD:h0@e0"
    )


def test_execute_remap_avoids_replan_increment_when_enabled():
    env = STS2CombatEnv()
    obs, info = env.reset(seed=7)
    combat = env.combat
    assert combat is not None
    mask = np.asarray(info["action_mask"])
    legal = set(legal_semantic_keys(combat, mask))
    play = stale = None
    for k in sorted(legal):
        if not k.startswith("play:"):
            continue
        prefix, target = k.split("@", 1)
        card = prefix.rsplit(":h", 1)[0]
        drifted = f"{card}:h99@{target}"
        if drifted not in legal and remap_plan_step_semantic_key(drifted, legal) == k:
            play, stale = k, drifted
            break
    assert play is not None and stale is not None

    rt = runtime_for_env(env)
    rt.player_turn_id = player_turn_id(combat)
    from sts2_env.eval.combat_turn_plan import living_enemy_intent_snapshot

    rt.plan = TurnPlanCandidate("plan_test", (stale, SEMANTIC_END_TURN))
    rt.step_index = 0
    rt.intent_snapshot = living_enemy_intent_snapshot(combat)
    rt.replans = 0
    tel = CombatJevTelemetry()

    class _Ppo:
        def predict(self, obs, action_masks=None, deterministic=True):
            return int(np.flatnonzero(np.asarray(action_masks) == 1)[0]), None

    action, shadow = choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=object(),
        combat_obs=encode_observation(combat),
        env=env,
        runtime=rt,
        telemetry=tel,
        execute_remap=True,
    )
    env.close()
    assert int(mask[action]) == 1
    assert shadow["turn_plan_failopen"] is False
    assert shadow["executed_id"] == play
    assert rt.replans == 0
    assert tel.turn_plan_replan_trigger.get("illegal_step", 0) == 0


def test_execute_remap_default_off_replans_on_drifted_slot():
    env = STS2CombatEnv()
    obs, info = env.reset(seed=7)
    combat = env.combat
    assert combat is not None
    mask = np.asarray(info["action_mask"])
    legal = set(legal_semantic_keys(combat, mask))
    stale = None
    for k in sorted(legal):
        if not k.startswith("play:"):
            continue
        prefix, target = k.split("@", 1)
        card = prefix.rsplit(":h", 1)[0]
        drifted = f"{card}:h99@{target}"
        if drifted not in legal and remap_plan_step_semantic_key(drifted, legal) == k:
            stale = drifted
            break
    assert stale is not None

    from sts2_env.eval.combat_turn_plan import living_enemy_intent_snapshot

    rt = runtime_for_env(env)
    rt.player_turn_id = player_turn_id(combat)
    rt.plan = TurnPlanCandidate("plan_test", (stale, SEMANTIC_END_TURN))
    rt.step_index = 0
    rt.intent_snapshot = living_enemy_intent_snapshot(combat)
    rt.replans = 0

    keys = legal_semantic_keys(combat, mask)
    board = serialize_combat_board_full(combat, mask)
    top, _, _, _ = cap_plans_for_turn_plan_choice(
        enumerate_candidate_plans(keys), board
    )

    class _Ppo:
        def predict(self, obs, action_masks=None, deterministic=True):
            return int(np.flatnonzero(np.asarray(action_masks) == 1)[0]), None

    class _Adapter:
        def system_one(self, state, questions):
            return {
                CHOICE_COMBAT_TURN_PLAN: JevAnswer(
                    status="ok", choice=top[0].plan_id, confidence=0.9
                )
            }

    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=_Adapter(),
        combat_obs=encode_observation(combat),
        env=env,
        runtime=rt,
        execute_remap=False,
    )
    env.close()
    assert rt.replans >= 1


def test_replan_trigger_telemetry_on_true_illegal_step():
    env = STS2CombatEnv()
    obs, info = env.reset(seed=1)
    combat = env.combat
    assert combat is not None
    mask = np.asarray(info["action_mask"])
    rt = runtime_for_env(env)
    rt.player_turn_id = player_turn_id(combat)
    rt.plan = TurnPlanCandidate(
        "plan_bad", ("play:NOT_A_REAL_CARD:h0@e0", SEMANTIC_END_TURN)
    )
    rt.step_index = 0
    rt.replans = 0
    tel = CombatJevTelemetry()

    class _Ppo:
        def predict(self, obs, action_masks=None, deterministic=True):
            return int(np.flatnonzero(np.asarray(action_masks) == 1)[0]), None

    keys = legal_semantic_keys(combat, mask)
    board = serialize_combat_board_full(combat, mask)
    top, _, _, _ = cap_plans_for_turn_plan_choice(
        enumerate_candidate_plans(keys), board
    )

    class _Adapter:
        def system_one(self, state, questions):
            return {
                CHOICE_COMBAT_TURN_PLAN: JevAnswer(
                    status="ok", choice=top[0].plan_id, confidence=0.9
                )
            }

    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(),
        adapter=_Adapter(),
        combat_obs=encode_observation(combat),
        env=env,
        runtime=rt,
        telemetry=tel,
    )
    env.close()
    assert rt.replans >= 1
    assert tel.turn_plan_replan_trigger.get("illegal_step", 0) >= 1
