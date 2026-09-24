"""HOLD turn-plan replay JSONL (fail + Boss episodes)."""
from __future__ import annotations

import json

import numpy as np

import sts2_env.cards  # noqa: F401
from sts2_env.eval.combat_hold import hold_jobs, run_hold_job_list
from sts2_env.eval.combat_turn_plan import (
    CHOICE_COMBAT_TURN_PLAN,
    choose_combat_turn_plan_action,
)
from sts2_env.eval.hold_turn_replay import (
    REPLAY_TURN_TRAJECTORY_KEYS,
    HoldTurnReplayWriter,
    should_retain_hold_turn_replay,
)
from sts2_env.eval.jev_types import JevAnswer

class _PlanAdapter:
    def __init__(self, plan_id: str):
        self.plan_id = plan_id

    def system_one(self, state, questions):
        return {
            CHOICE_COMBAT_TURN_PLAN: JevAnswer(
                status="ok", choice=self.plan_id, confidence=0.9
            )
        }


class _Ppo:
    def predict(self, obs, action_masks=None, deterministic=True):
        valid = np.flatnonzero(np.asarray(action_masks) == 1)
        return int(valid[0]), None


def test_record_plan_choice_includes_trajectory_keys():
    from sts2_env.eval.combat_turn_plan import TurnPlanCandidate
    from sts2_env.eval.hold_turn_replay import HoldTurnPlanEpisodeReplay

    board = {
        "self": {"hp": 42, "max_hp": 70, "block": 0, "energy": 3, "hand": []},
        "enemies": [{"name": "Slime", "slot": 0, "hp": 8, "max_hp": 20, "block": 0, "intent": "?"}],
        "turn": {},
        "piles": {},
    }
    rec = HoldTurnPlanEpisodeReplay(1, 0, 19, "boss", 0)
    rec.record_plan_choice(
        player_turn=0,
        plans=[TurnPlanCandidate("plan_0000", ("end_turn",))],
        board=board,
        pruned_plan_count=0,
        picked_plan_id="plan_0000",
        pick_error=None,
        bh_assist=None,
        replan_count=1,
        replan_cap_hit=False,
        fail_open=False,
        fail_open_reason=None,
    )
    rec.patch_last_turn_trajectory(
        {**board, "self": {**board["self"], "hp": 40}},
        replan_count=1,
    )
    turn = rec.turns[0]
    for key in REPLAY_TURN_TRAJECTORY_KEYS:
        assert key in turn
    assert turn["player_hp_start"] == 42
    assert turn["player_hp_end"] == 40
    assert turn["enemies_hp"][0]["hp"] == 8


def test_record_replan_cap_catastrophe_sets_hit_flag():
    from sts2_env.eval.hold_turn_replay import HoldTurnPlanEpisodeReplay

    board = {
        "self": {"hp": 20, "max_hp": 70, "block": 0, "energy": 3, "hand": []},
        "enemies": [],
        "turn": {},
        "piles": {},
    }
    rec = HoldTurnPlanEpisodeReplay(1, 0, 19, "boss", 0)
    rec.record_replan_cap_catastrophe(
        player_turn=2,
        board=board,
        replan_count=4,
        shadow={"turn_plan_failopen_reason": "cap_exceeded"},
    )
    turn = rec.turns[0]
    assert turn["replan_cap_hit"] is True
    assert turn["replan_count"] == 4
    assert turn["fail_open"] is True
    assert turn["fail_open_reason"] == "cap_exceeded"


def test_choose_combat_turn_plan_records_replan_cap_on_runtime():
    from sts2_env.eval.combat_turn_plan import (
        runtime_for_env,
        choose_combat_turn_plan_action,
        player_turn_id,
    )
    from sts2_env.eval.hold_turn_replay import (
        HOLD_TURN_REPLAY_ENV_ATTR,
        HoldTurnPlanEpisodeReplay,
    )
    from sts2_env.gym_env.observation import encode_observation

    env, combat, mask = _reset()
    obs = encode_observation(combat)
    rt = runtime_for_env(env)
    rt.player_turn_id = player_turn_id(combat)
    rt.replans = 4
    rec = HoldTurnPlanEpisodeReplay(0, 0, 16, "elite", 0)
    setattr(env, HOLD_TURN_REPLAY_ENV_ATTR, rec)
    ppo = _Ppo()
    choose_combat_turn_plan_action(
        combat,
        mask,
        np.random.RandomState(0),
        ppo,
        adapter=_PlanAdapter("plan_0000"),
        combat_obs=obs,
        env=env,
        runtime=rt,
    )
    env.close()
    assert rec.turns
    cap_rows = [t for t in rec.turns if t.get("replan_cap_hit")]
    assert cap_rows
    assert cap_rows[-1]["replan_count"] >= 4


def _reset():
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    env = STS2CombatEnv()
    obs, info = env.reset(seed=0)
    combat = env.combat
    mask = info.get("action_mask")
    if mask is None:
        mask = env.action_masks()
    return env, combat, mask


def test_should_retain_fail_and_boss_only():
    assert should_retain_hold_turn_replay(win=False, bucket="elite")
    assert should_retain_hold_turn_replay(win=True, bucket="boss")
    assert not should_retain_hold_turn_replay(win=True, bucket="elite")


def test_hold_turn_replay_boss_fail_episode_jsonl(tmp_path):
    jobs = hold_jobs(n_eps=1)
    boss_job = next(j for j in jobs if j["bucket"] == "boss")
    replay_path = tmp_path / "hold_turn_replay_w0.jsonl"
    writer = HoldTurnReplayWriter(replay_path)
    from sts2_env.eval.combat_jev import CombatJevTelemetry

    ppo = _Ppo()
    adapter = _PlanAdapter("plan_not_in_list")
    telemetry = CombatJevTelemetry()

    def choose_fn(env, obs, mask):
        combat = getattr(env, "combat", None)
        if combat is None:
            return int(ppo.predict(obs, action_masks=mask)[0])
        local, _shadow = choose_combat_turn_plan_action(
            combat,
            mask,
            np.random.RandomState(0),
            ppo,
            adapter=adapter,
            combat_obs=obs,
            env=env,
            telemetry=telemetry,
        )
        return int(local)

    rows = run_hold_job_list(
        [boss_job],
        lambda o, m: int(ppo.predict(o, action_masks=m)[0]),
        choose_fn=choose_fn,
        max_steps=400,
        turn_replay_writer=writer,
        turn_replay_meta={"combat_policy": "jev-turn", "bh_assist": "off"},
        combat_jev_telemetry=telemetry,
    )
    assert rows
    assert rows[0].get("had_turn_plan_catastrophe") is True
    assert int(rows[0].get("turn_plan_catastrophe_turns") or 0) >= 1
    assert replay_path.is_file()
    line = replay_path.read_text(encoding="utf-8").strip().splitlines()[0]
    doc = json.loads(line)
    assert doc["seed"] == boss_job["seed"]
    assert doc["enc_id"] == boss_job["enc_id"]
    assert doc["fixture_index"] == boss_job["fixture_index"]
    assert doc["bucket"] == "boss"
    assert doc["turns"]
    turn0 = doc["turns"][0]
    for key in REPLAY_TURN_TRAJECTORY_KEYS:
        assert key in turn0, key
    assert isinstance(turn0["enemies_hp"], list)
    assert turn0["fail_open"] is True
    assert turn0["fail_open_reason"] == "illegal_plan"
    assert turn0["plan_shortlist"]
    assert turn0["plan_shortlist"][0]["plan_id"]
    assert turn0["plan_shortlist"][0]["criteria"]
    assert turn0["choice_pick"]["error"] == "illegal_plan"
    assert "terminal" in doc
    assert "player_hp" in doc["terminal"]
    assert "deck_summary" in doc["terminal"]
