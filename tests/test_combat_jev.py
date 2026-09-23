"""Combat-Jev bypass contract: combat_step_choice, conf 0.45, fail-open to bh_v1."""
from __future__ import annotations

import numpy as np
import pytest

from sts2_env.eval.combat_jev import (
    CHOICE_COMBAT_STEP,
    COMBAT_JEV_CONF_MIN,
    COMBAT_JEV_INSTRUCTIONS,
    CombatJevTelemetry,
    FAILOPEN_BAD_ID,
    FAILOPEN_EMPTY_LIST,
    FAILOPEN_ERROR,
    FAILOPEN_LOW_CONF,
    FAILOPEN_TIMEOUT,
    choose_combat_step,
    classify_jev_error,
    combat_action_id,
    compress_combat_state,
    enumerate_legal_combat_actions,
)
from sts2_env.eval.jev_client import StubJevClient
from sts2_env.eval.jev_types import JevAnswer, JevError
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import OBS_SIZE


class _Ppo:
    def __init__(self, action: int):
        self.action = int(action)
        self.n = 0
        self.seen_widths: list[int] = []

    def predict(self, obs, action_masks=None, deterministic=True):
        self.n += 1
        arr = np.asarray(obs)
        self.seen_widths.append(int(arr.shape[-1]))
        return self.action, None


class _PickAdapter:
    def __init__(self, choice: str | None, conf: float | None = 0.9):
        self.choice = choice
        self.conf = conf
        self.n = 0
        self.last_questions: dict | None = None
        self.last_state: dict | None = None

    def system_one(self, state, questions):
        self.n += 1
        self.last_state = state
        self.last_questions = questions
        return {
            name: JevAnswer(status="ok", choice=self.choice, confidence=self.conf)
            for name in questions
        }


class _ErrorAdapter:
    def __init__(self, msg: str = "forced"):
        self.msg = msg
        self.n = 0

    def system_one(self, state, questions):
        self.n += 1
        raise JevError(self.msg)


def _reset_combat(seed: int = 1):
    env = STS2CombatEnv()
    obs, info = env.reset(seed=seed)
    mask = np.asarray(info["action_mask"])
    combat = env.combat
    assert combat is not None
    return env, combat, obs, mask


def test_contract_constants_locked():
    assert CHOICE_COMBAT_STEP == "combat_step_choice"
    assert COMBAT_JEV_CONF_MIN == 0.45
    assert "legal shortlist" in COMBAT_JEV_INSTRUCTIONS
    assert "intent damage" in COMBAT_JEV_INSTRUCTIONS
    assert "fail-open" in COMBAT_JEV_INSTRUCTIONS
    assert "hard-pick" in COMBAT_JEV_INSTRUCTIONS


def test_enumerate_legal_is_play_potion_end_only():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    env.close()
    assert options
    ids = {action_id for action_id, _idx, _sum in options}
    assert combat_action_id(0) in ids
    for action_id, idx, summary in options:
        assert action_id == combat_action_id(idx)
        assert int(mask[idx]) == 1
        assert (
            summary.startswith("end turn |")
            or summary.startswith("play ")
            or summary.startswith("use potion ")
        )
        assert "hp=" in summary
        assert "energy=" in summary
        assert "end_turn=" in summary
        assert "intent:" in summary


def test_choice_question_includes_tightened_instructions():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    from sts2_env.eval.combat_jev import _choice_question

    q = _choice_question(options)
    assert q["instructions"] == COMBAT_JEV_INSTRUCTIONS
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {action_id for action_id, _idx, _ in options}
    for _aid, _idx, summary in options:
        assert q["criteria"][_aid] == summary
    env.close()


def test_choose_success_uses_shortlist_id_not_ppo():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    jev_id, jev_idx, _ = options[-1]
    ppo_idx = int(options[0][1])
    assert jev_idx != ppo_idx
    tel = CombatJevTelemetry()
    adapter = _PickAdapter(jev_id, conf=0.9)
    local, shadow = choose_combat_step(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(ppo_idx),
        adapter=adapter,
        combat_obs=obs,
        telemetry=tel,
    )
    env.close()
    assert local == jev_idx
    assert adapter.n == 1
    assert CHOICE_COMBAT_STEP in adapter.last_questions
    assert adapter.last_questions[CHOICE_COMBAT_STEP]["type"] == "choice"
    assert "self" in adapter.last_state
    assert "enemies" in adapter.last_state
    assert "turn" in adapter.last_state
    assert shadow["combat_jev_failopen"] is False
    assert tel.calls == 1
    assert tel.fail_open == 0
    assert shadow["shadow_suggestion"] == jev_id


def test_low_conf_and_missing_conf_fail_open_to_ppo():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    jev_id, jev_idx, _ = options[-1]
    ppo_idx = int(options[0][1])
    assert jev_idx != ppo_idx
    rng = np.random.RandomState(0)
    tel = CombatJevTelemetry()
    local, shadow = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(ppo_idx),
        adapter=_PickAdapter(jev_id, conf=0.34),
        combat_obs=obs,
        telemetry=tel,
    )
    assert local == ppo_idx
    assert shadow["jev_failopen_reason"] == FAILOPEN_LOW_CONF
    local2, shadow2 = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(ppo_idx),
        adapter=_PickAdapter(jev_id, conf=None),
        combat_obs=obs,
        telemetry=tel,
    )
    env.close()
    assert local2 == ppo_idx
    assert shadow2["jev_failopen_reason"] == FAILOPEN_LOW_CONF
    assert tel.failopen_reason[FAILOPEN_LOW_CONF] == 2


def test_conf_at_threshold_is_accepted():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    jev_id, jev_idx, _ = options[-1]
    ppo_idx = int(options[0][1])
    local, shadow = choose_combat_step(
        combat,
        mask,
        np.random.RandomState(0),
        _Ppo(ppo_idx),
        adapter=_PickAdapter(jev_id, conf=COMBAT_JEV_CONF_MIN),
        combat_obs=obs,
    )
    env.close()
    assert local == jev_idx
    assert shadow["combat_jev_failopen"] is False


def test_bad_id_and_stub_and_error_fail_open():
    env, combat, obs, mask = _reset_combat()
    options = enumerate_legal_combat_actions(combat, mask)
    ppo_idx = int(options[0][1])
    rng = np.random.RandomState(0)
    tel = CombatJevTelemetry()
    local, shadow = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(ppo_idx),
        adapter=_PickAdapter("invented_off_list", conf=0.99),
        combat_obs=obs,
        telemetry=tel,
    )
    assert local == ppo_idx
    assert shadow["jev_failopen_reason"] == FAILOPEN_BAD_ID
    local2, shadow2 = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(ppo_idx),
        adapter=StubJevClient(),
        combat_obs=obs,
        telemetry=tel,
    )
    assert local2 == ppo_idx
    assert shadow2["jev_failopen_reason"] == FAILOPEN_BAD_ID
    local3, shadow3 = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(ppo_idx),
        adapter=_ErrorAdapter("forced"),
        combat_obs=obs,
        telemetry=tel,
    )
    env.close()
    assert local3 == ppo_idx
    assert shadow3["jev_failopen_reason"] == FAILOPEN_ERROR
    assert tel.fail_open == 3


def test_timeout_reason_and_ppo_unavailable_random():
    assert classify_jev_error(JevError("typesafe network error: timed out")) == FAILOPEN_TIMEOUT
    env, combat, obs, mask = _reset_combat()
    tel = CombatJevTelemetry()
    rng = np.random.RandomState(0)
    local, shadow = choose_combat_step(
        combat,
        mask,
        rng,
        _Ppo(0),
        adapter=_ErrorAdapter("urlopen timeout"),
        combat_obs=obs,
        telemetry=tel,
    )
    assert shadow["jev_failopen_reason"] == FAILOPEN_TIMEOUT
    valid = set(np.flatnonzero(mask == 1).tolist())
    local2, _ = choose_combat_step(
        combat,
        mask,
        rng,
        None,
        adapter=_ErrorAdapter("forced"),
        combat_obs=obs,
        telemetry=tel,
    )
    env.close()
    assert int(local2) in valid
    assert tel.failopen_reason[FAILOPEN_TIMEOUT] == 1
    assert tel.failopen_reason[FAILOPEN_ERROR] == 1


def test_compressed_state_truncates_and_report_schema():
    env, combat, obs, mask = _reset_combat()
    state = compress_combat_state(combat, end_turn_legal=True)
    env.close()
    assert set(state) >= {"self", "enemies", "turn", "truncated"}
    assert "hp" in state["self"] and "max_hp" in state["self"]
    assert "energy" in state["self"]
    assert "hand" in state["self"]
    assert len(state["self"]["hand"]) <= 8
    assert len(state["enemies"]) <= 3
    assert "player_turn_index" in state["turn"]
    tel = CombatJevTelemetry()
    tel.record(10.0)
    tel.record(20.0, fail_reason=FAILOPEN_LOW_CONF)
    report = tel.as_report(n_episodes=2)
    assert report["jev_calls"] == 2
    assert report["jev_failopen"] == 1
    assert report["failopen_rate"] == 0.5
    assert report["latency_ms"]["p50"] is not None
    assert report["latency_ms"]["p95"] is not None
    assert report["combat_jev_calls_mean"] == 1.0
    assert set(report["jev_failopen_reason"]) == {
        FAILOPEN_TIMEOUT,
        FAILOPEN_ERROR,
        FAILOPEN_BAD_ID,
        FAILOPEN_LOW_CONF,
        FAILOPEN_EMPTY_LIST,
    }
    assert report["jev_fulfilled_rate"] == 0.0
    assert report["catastrophe_failopen_rate"] == 0.0
    assert report["turn_plan_turns"] == 0


def test_obs_width_stays_combat():
    env, combat, obs, mask = _reset_combat()
    ppo = _Ppo(0)
    choose_combat_step(
        combat,
        mask,
        np.random.RandomState(0),
        ppo,
        adapter=StubJevClient(),
        combat_obs=obs,
    )
    env.close()
    assert ppo.seen_widths == [OBS_SIZE]


def test_combat_jev_telemetry_merge_and_lazy_eval_attr():
    a = CombatJevTelemetry()
    a.record(10.0)
    a.record(20.0, fail_reason=FAILOPEN_LOW_CONF)
    b = CombatJevTelemetry()
    b.record(30.0, fail_reason=FAILOPEN_TIMEOUT)
    merged = CombatJevTelemetry.from_dict(a.to_dict())
    merged.merge(b)
    assert merged.calls == 3
    assert merged.fail_open == 2
    assert merged.failopen_reason[FAILOPEN_LOW_CONF] == 1
    assert merged.failopen_reason[FAILOPEN_TIMEOUT] == 1
    report = merged.as_report(n_episodes=3)
    assert report["jev_calls"] == 3
    assert report["jev_failopen"] == 2

    merged.record_turn_plan_turn(fulfilled=True)
    merged.record_turn_plan_turn(fulfilled=False, catastrophe_reason=FAILOPEN_TIMEOUT)
    merged.record_turn_plan_turn(fulfilled=True)
    roundtrip = CombatJevTelemetry.from_dict(merged.to_dict())
    assert roundtrip.turn_plan_turns == 3
    assert roundtrip.turn_plan_fulfilled == 2
    assert roundtrip.turn_plan_catastrophe == 1

    import sts2_env.eval as eval_pkg

    lazy = eval_pkg.CombatJevTelemetry
    assert lazy is CombatJevTelemetry
