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


def test_should_retain_fail_and_boss_only():
    assert should_retain_hold_turn_replay(win=False, bucket="elite")
    assert should_retain_hold_turn_replay(win=True, bucket="boss")
    assert not should_retain_hold_turn_replay(win=True, bucket="elite")


def test_hold_turn_replay_boss_fail_episode_jsonl(tmp_path):
    jobs = hold_jobs(n_eps=1)
    boss_job = next(j for j in jobs if j["bucket"] == "boss")
    replay_path = tmp_path / "hold_turn_replay_w0.jsonl"
    writer = HoldTurnReplayWriter(replay_path)
    ppo = _Ppo()
    adapter = _PlanAdapter("plan_not_in_list")

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
        )
        return int(local)

    rows = run_hold_job_list(
        [boss_job],
        lambda o, m: int(ppo.predict(o, action_masks=m)[0]),
        choose_fn=choose_fn,
        max_steps=400,
        turn_replay_writer=writer,
        turn_replay_meta={"combat_policy": "jev-turn", "bh_assist": "off"},
    )
    assert rows
    assert replay_path.is_file()
    line = replay_path.read_text(encoding="utf-8").strip().splitlines()[0]
    doc = json.loads(line)
    assert doc["seed"] == boss_job["seed"]
    assert doc["enc_id"] == boss_job["enc_id"]
    assert doc["fixture_index"] == boss_job["fixture_index"]
    assert doc["bucket"] == "boss"
    assert doc["turns"]
    turn0 = doc["turns"][0]
    assert turn0["plan_shortlist"]
    assert turn0["plan_shortlist"][0]["plan_id"]
    assert turn0["plan_shortlist"][0]["criteria"]
    assert turn0["choice_pick"]["error"] == "illegal_plan"
    assert "terminal" in doc
    assert "player_hp" in doc["terminal"]
    assert "deck_summary" in doc["terminal"]
