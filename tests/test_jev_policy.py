"""Jev non-combat policy: confidence, hp_pressure, error fallback, plus-card."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sts2_env.eval.jev import (
    CARD_FIT_ASSIST_MIN,
    CARD_FIT_ASSIST_REASON,
    CHOICE_CONFIDENCE_MIN,
    CONTENT_MAP_REF,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
    JevAnswer,
    JevError,
    LiveJevClient,
    apply_choice_confidence,
    build_jev_adapter,
    local_hp_pressure,
    rest_or_continue_override,
)
from sts2_env.eval.jev_policy import (
    classify_decision,
    choose_jev_noncombat,
    collect_candidates,
    is_potion_or_relic_reward,
    strip_illegal_invisible,
)
from sts2_env.gym_env.run_env import (
    TOTAL_ACTIONS,
    _CARD_RWD_EXTRA_START,
    _CARD_RWD_START,
    _MAP_START,
)
from sts2_env.run.run_manager import RunManager


class ScriptedJev:
    """Deterministic adapter: queued answers, or a raised JevError."""

    def __init__(self, call_answers=None, error=None):
        self.call_answers = list(call_answers or [])
        self.error = error
        self.calls: list[dict] = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.error:
            raise JevError(self.error)
        payload = self.call_answers.pop(0) if self.call_answers else {}
        out = {}
        for name in questions:
            spec = payload.get(name, {})
            out[name] = JevAnswer(
                status=spec.get("status", "ok"),
                choice=spec.get("choice"),
                confidence=spec.get("confidence"),
                score=spec.get("score"),
            )
        return out


def _player(hp=80, max_hp=80):
    return SimpleNamespace(current_hp=hp, max_hp=max_hp, gold=99, deck=[1, 2, 3])


def _env(phase, actions, hp=80, max_hp=80):
    rs = SimpleNamespace(
        player=_player(hp, max_hp),
        current_act_index=0,
        total_floor=3,
        relics=[],
    )
    mgr = SimpleNamespace(
        phase=phase,
        get_available_actions=lambda: actions,
        run_state=rs,
    )
    return SimpleNamespace(_mgr=mgr)


def _map_env(nodes, hp=80, max_hp=80, illegal=()):
    actions = []
    for ptype, coord in nodes:
        actions.append({"action": "move", "coord": coord, "point_type": ptype})
    env = _env(RunManager.PHASE_MAP_CHOICE, actions, hp=hp, max_hp=max_hp)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    for i, (ptype, _) in enumerate(nodes):
        if i in illegal:
            continue
        mask[_MAP_START + i] = 1
    return env, mask


def test_choice_confidence_threshold():
    ok = apply_choice_confidence(JevAnswer(status="ok", choice="a", confidence=0.65))
    assert ok.status == "ok"
    low = apply_choice_confidence(JevAnswer(status="ok", choice="a", confidence=0.64))
    assert low.status == "uncertain"
    assert "0.64" in (low.fallback_reason or "")
    none = apply_choice_confidence(JevAnswer(status="ok", choice="a", confidence=None))
    assert none.status == "uncertain"
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_hp_pressure_thresholds_and_local_formula():
    assert rest_or_continue_override(2.0) == "rest"
    assert rest_or_continue_override(2.5) == "rest"
    assert rest_or_continue_override(1.0) == "continue"
    assert rest_or_continue_override(0.5) == "continue"
    assert rest_or_continue_override(1.5) is None
    assert local_hp_pressure(80, 80) == 1.0
    assert local_hp_pressure(40, 80) == 2.0


def test_strip_unassigned_and_illegal_before_choice():
    env, mask = _map_env(
        [
            ("MONSTER", (0, 1)),
            ("UNASSIGNED", (1, 1)),
            ("REST_SITE", (2, 1)),
        ],
        illegal={2},
    )
    all_cands = collect_candidates(env, mask)
    kept = strip_illegal_invisible(all_cands)
    keys = {c.key for c in kept}
    assert keys == {"map_0"}
    assert classify_decision(env._mgr.phase, kept) == "map_fork"
    adapter = ScriptedJev([{"pick": {"choice": "map_0", "confidence": 0.9}}])
    _, _ = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    criteria = adapter.calls[0]["questions"]["pick"]["criteria"]
    assert set(criteria) == {"map_0"}


def test_unknown_map_node_is_legal_visible():
    env, mask = _map_env([("UNKNOWN", (0, 1)), ("MONSTER", (1, 1))])
    kept = strip_illegal_invisible(collect_candidates(env, mask))
    assert {c.key for c in kept} == {"map_0", "map_1"}


def test_map_fork_respects_confident_choice():
    env, mask = _map_env([("MONSTER", (0, 1)), ("ELITE", (1, 1))])
    adapter = ScriptedJev(
        [{"pick": {"choice": "map_1", "confidence": 0.9}}]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_decision"] == "map_fork"
    assert log["shadow_status"] == "ok"
    assert log["shadow_suggestion"] == "map_1"
    assert log["shadow_confidence"] == 0.9
    assert action == _MAP_START + 1
    criteria = adapter.calls[0]["questions"]["pick"]["criteria"]
    assert "map_0" in criteria and "map_1" in criteria
    assert CONTENT_MAP_REF in adapter.calls[0]["questions"]["pick"]["instructions"]


def test_low_confidence_falls_back_to_legal_random():
    env, mask = _map_env([("MONSTER", (0, 1)), ("ELITE", (1, 1))])
    adapter = ScriptedJev(
        [{"pick": {"choice": "map_1", "confidence": 0.64}}]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_status"] == "uncertain"
    assert log["shadow_suggestion"] == "map_1"
    assert log["shadow_confidence"] == 0.64
    assert log["shadow_fallback_reason"]
    assert mask[action] == 1
    assert action in {_MAP_START, _MAP_START + 1}


def test_api_error_logs_error_and_still_acts():
    env, mask = _map_env([("MONSTER", (0, 1)), ("ELITE", (1, 1))])
    adapter = ScriptedJev(error="typesafe HTTP 500: boom")
    rng = np.random.RandomState(1)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_status"] == "error"
    assert "500" in (log["shadow_fallback_reason"] or "")
    assert mask[action] == 1


def test_rest_or_continue_hp_pressure_prefers_rest():
    env, mask = _map_env(
        [("REST_SITE", (0, 1)), ("MONSTER", (1, 1))],
        hp=20,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 2.4},
                "pick": {"choice": "map_1", "confidence": 0.99},
            }
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_decision"] == "rest_or_continue"
    assert action == _MAP_START  # rest
    assert log["shadow_suggestion"] == "map_0"
    assert log["shadow_hp_pressure"] == pytest.approx(2.4)
    assert "prefer rest" in (log["shadow_fallback_reason"] or "")


def test_rest_or_continue_hp_pressure_prefers_continue():
    env, mask = _map_env(
        [("REST_SITE", (0, 1)), ("MONSTER", (1, 1))],
        hp=80,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 0.4},
                "pick": {"choice": "map_0", "confidence": 0.99},
            },
            {"pick": {"choice": "map_1", "confidence": 0.9}},
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_decision"] == "rest_or_continue"
    assert action == _MAP_START + 1
    assert "prefer continue" in (log["shadow_fallback_reason"] or "")
    assert log["shadow_hp_pressure"] == pytest.approx(0.4)


def test_rest_or_continue_middle_trusts_choice():
    env, mask = _map_env(
        [("REST_SITE", (0, 1)), ("MONSTER", (1, 1))],
        hp=50,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 1.4},
                "pick": {"choice": "map_1", "confidence": 0.8},
            }
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert action == _MAP_START + 1
    assert log["shadow_status"] == "ok"
    assert log["shadow_suggestion"] == "map_1"
    assert log["shadow_hp_pressure"] == pytest.approx(1.4)


def test_plus_card_criterion_in_card_reward_choice():
    actions = [
        {"action": "pick_card", "card_id": "STRIKE", "upgraded": False},
        {"action": "pick_card", "card_id": "BASH", "upgraded": True},
    ]
    env = _env(RunManager.PHASE_CARD_REWARD, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_CARD_RWD_START] = 1
    mask[_CARD_RWD_START + 1] = 1
    mask[_CARD_RWD_START + 3] = 1  # skip
    adapter = ScriptedJev(
        [{"pick": {"choice": "card_0", "confidence": 0.9}}]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_decision"] == "card_reward"
    assert action == _CARD_RWD_START
    q = adapter.calls[0]["questions"]["pick"]
    assert PLUS_CARD_CRITERION in q["instructions"]
    assert CONTENT_MAP_REF in q["instructions"]
    assert "Neow+early" in q["instructions"]
    assert "mid" in q["instructions"].lower()
    assert NEOW_EARLY_CARD_INSTRUCTIONS in q["instructions"] or "Neow+early" in q["instructions"]
    assert PLUS_CARD_CRITERION in q["criteria"]["card_1"]
    assert PLUS_CARD_CRITERION not in q["criteria"]["card_0"]
    assert "card_fit" in adapter.calls[0]["questions"]
    assert adapter.calls[0]["questions"]["card_fit"]["type"] == "score"
    assert "card_skip" not in q["criteria"]
    assert "Anger" in q["criteria"]["card_0"] or "ANGER" in q["criteria"]["card_0"] or "anger" in q["criteria"]["card_0"].lower() or "Pick card 0" in q["criteria"]["card_0"]


def test_card_reward_skip_only_when_action_is_skip():
    actions = [
        {"action": "pick_card", "card_id": "ANGER", "upgraded": False, "rarity": "COMMON"},
        {"action": "pick_card", "card_id": "INFLAME", "upgraded": False, "rarity": "UNCOMMON"},
        {"action": "pick_card", "card_id": "BASH", "upgraded": True, "rarity": "BASIC"},
        {"action": "pick_card", "card_id": "HEMOKINESIS", "upgraded": False, "rarity": "UNCOMMON"},
    ]
    env = _env(RunManager.PHASE_CARD_REWARD, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    for i in range(3):
        mask[_CARD_RWD_START + i] = 1
    mask[_CARD_RWD_EXTRA_START] = 1
    cands = collect_candidates(env, mask)
    keys = [c.key for c in cands]
    assert keys == ["card_0", "card_1", "card_2", "card_3"]
    assert cands[3].run_action == _CARD_RWD_EXTRA_START
    assert "Smith/Neow" in cands[2].description
    env2 = _env(
        RunManager.PHASE_CARD_REWARD,
        actions + [{"action": "skip"}],
    )
    with_skip = collect_candidates(env2, mask)
    assert any(c.key == "card_skip" for c in with_skip)


def test_potion_or_relic_reward_does_not_call_card_jev():
    for action_name in ("pick_potion", "pick_relic_reward"):
        actions = [
            {"action": action_name, "potion_id": "FAKE"}
            if action_name == "pick_potion"
            else {"action": action_name, "relic_id": "FAKE"},
            {"action": "skip_potion" if action_name == "pick_potion" else "skip_relic"},
        ]
        assert is_potion_or_relic_reward(actions)
        env = _env(RunManager.PHASE_CARD_REWARD, actions)
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        mask[_CARD_RWD_START] = 1
        mask[_CARD_RWD_START + 3] = 1
        adapter = ScriptedJev([{"pick": {"choice": "card_0", "confidence": 0.99}}])
        rng = np.random.RandomState(0)
        action, log = choose_jev_noncombat(env, mask, rng, adapter)
        assert adapter.calls == []
        assert log["shadow_decision"] == "potion_or_relic_reward"
        assert log["shadow_fallback_reason"] == POTION_OR_RELIC_REASON
        assert log["shadow_status"] != "ok"
        assert mask[action] == 1


def test_card_fit_assist_lands_when_choice_not_skip():
    actions = [
        {"action": "pick_card", "card_id": "ANGER", "upgraded": False, "rarity": "COMMON"},
        {"action": "pick_card", "card_id": "INFLAME", "upgraded": False, "rarity": "UNCOMMON"},
        {"action": "skip"},
    ]
    env = _env(RunManager.PHASE_CARD_REWARD, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_CARD_RWD_START] = 1
    mask[_CARD_RWD_START + 1] = 1
    mask[_CARD_RWD_START + 3] = 1
    adapter = ScriptedJev(
        [
            {
                "pick": {"choice": "card_1", "confidence": 0.50},
                "card_fit": {"score": 2.0},
            }
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert action == _CARD_RWD_START + 1
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] == CARD_FIT_ASSIST_REASON
    assert log["jev_card_fit"] == pytest.approx(2.0)
    assert CARD_FIT_ASSIST_MIN == 2.0
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_card_fit_assist_does_not_land_skip():
    actions = [
        {"action": "pick_card", "card_id": "ANGER", "upgraded": False, "rarity": "COMMON"},
        {"action": "skip"},
    ]
    env = _env(RunManager.PHASE_CARD_REWARD, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_CARD_RWD_START] = 1
    mask[_CARD_RWD_START + 3] = 1
    adapter = ScriptedJev(
        [
            {
                "pick": {"choice": "card_skip", "confidence": 0.40},
                "card_fit": {"score": 3.0},
            }
        ]
    )
    rng = np.random.RandomState(1)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_status"] == "uncertain"
    assert log["shadow_fallback_reason"] != CARD_FIT_ASSIST_REASON
    assert mask[action] == 1


def test_card_fit_below_assist_stays_random():
    actions = [
        {"action": "pick_card", "card_id": "ANGER", "upgraded": False},
        {"action": "pick_card", "card_id": "INFLAME", "upgraded": False},
    ]
    env = _env(RunManager.PHASE_CARD_REWARD, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_CARD_RWD_START] = 1
    mask[_CARD_RWD_START + 1] = 1
    adapter = ScriptedJev(
        [
            {
                "pick": {"choice": "card_1", "confidence": 0.50},
                "card_fit": {"score": 1.0},
            }
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_status"] == "uncertain"
    assert log.get("jev_card_fit") == pytest.approx(1.0)
    assert mask[action] == 1
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_live_client_missing_key_raises_jev_error():
    client = LiveJevClient(api_key="")
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        client.system_one({}, {"pick": {"type": "choice", "criteria": {"a": "x"}}})


def test_build_adapter_stub_vs_live():
    stub = build_jev_adapter(enabled=False)
    answers = stub.system_one({}, {"pick": {"type": "choice"}})
    assert answers["pick"].status == "stub"
    live = build_jev_adapter(enabled=True)
    assert isinstance(live, LiveJevClient)


def test_content_map_ref_is_the_stub_doc():
    root = Path(__file__).resolve().parents[1]
    assert CONTENT_MAP_REF == "docs/act1_content_map.md"
    text = (root / CONTENT_MAP_REF).read_text()
    assert "not a natural" in text.lower() or "NOT natural" in text
    assert "Smith" in text and "Neow" in text
    assert "TODO(Surplus/Jev)" in text
    assert "UNASSIGNED" in text
    assert "UNKNOWN" in text
    assert "card_fit" in text
    assert "pick_potion" in text
    assert "Neow+early" in text
