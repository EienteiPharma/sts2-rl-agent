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
    CHOICE_EVENT,
    CHOICE_NEOW_BOON,
    CONTENT_MAP_REF,
    DEFAULT_JEV_PHASES,
    EVENT_OPTIONS_EMPTY_REASON,
    HP_PRESSURE_ASSIST_REASON,
    JEV_EVENT_OFF_REASON,
    JEV_NEOW_OFF_REASON,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    NEOW_OPTIONS_EMPTY_REASON,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
    REST_CHOICE_MIN_CONFIDENCE,
    REST_HEAL_ASSIST_CONF,
    REST_SMITH_ASSIST_CONF,
    SHOP_RANDOM_REASON,
    SMITH_ASSIST_REASON,
    UNKNOWN_DEFER_CONF,
    UNKNOWN_DEFERRED_REASON,
    JevAnswer,
    JevError,
    LiveJevClient,
    apply_choice_confidence,
    build_jev_adapter,
    local_hp_pressure,
    rest_or_continue_override,
)
from sts2_env.eval.jev_policy import (
    JevPolicyFlags,
    build_event_options,
    build_options,
    build_rest_options,
    classify_decision,
    choose_jev_noncombat,
    collect_candidates,
    is_potion_or_relic_reward,
    resolve_jev_flags,
    strip_illegal_invisible,
)
from sts2_env.gym_env.run_env import (
    STS2RunEnv,
    TOTAL_ACTIONS,
    _CARD_RWD_EXTRA_START,
    _CARD_RWD_START,
    _COMBAT_START,
    _EVENT_START,
    _MAP_START,
    _REST_START,
    _SHOP_START,
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
    assert "hp_pressure" not in adapter.calls[0]["questions"]


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
    assert "unknown_deferred" in text
    assert "待核" in text
    assert "neow_boon" in text
    assert "event_choice" in text
    assert "event_options_empty" in text
    assert CHOICE_CONFIDENCE_MIN == 0.65
    assert UNKNOWN_DEFER_CONF == 0.80
    assert "event" not in DEFAULT_JEV_PHASES


def _event_env(actions, event_id="BrainLeech", hp=80, max_hp=80):
    env = _env(RunManager.PHASE_EVENT, actions, hp=hp, max_hp=max_hp)
    env._mgr._event_model = SimpleNamespace(event_id=event_id, pending_choice=None)
    return env


def _event_mask(n_event=2, pending=False, n_choose=0):
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    if pending:
        mask[_COMBAT_START] = 1
        for i in range(n_choose):
            mask[_COMBAT_START + 1 + i] = 1
        return mask
    for i in range(n_event):
        mask[_EVENT_START + i] = 1
    return mask


EVENT_ON = JevPolicyFlags(phases=frozenset({"map", "rest", "card", "event"}), event=True)


def test_unknown_deferred_when_pressure_high_and_conf_shy():
    env, mask = _map_env(
        [("UNKNOWN", (0, 1)), ("MONSTER", (1, 1))],
        hp=20,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 2.5},
                "pick": {"choice": "map_0", "confidence": 0.70},
            }
        ]
    )
    rng = np.random.RandomState(0)
    action, log = choose_jev_noncombat(env, mask, rng, adapter)
    assert log["shadow_fallback_reason"] == UNKNOWN_DEFERRED_REASON
    assert log["shadow_status"] != "ok"
    assert log["shadow_suggestion"] == "map_0"
    assert log["shadow_confidence"] == 0.70
    assert log["shadow_hp_pressure"] == pytest.approx(2.5)
    assert action == _MAP_START + 1  # non-Unknown pool only
    assert CHOICE_CONFIDENCE_MIN == 0.65
    assert UNKNOWN_DEFER_CONF == 0.80
    assert "hp_pressure" in adapter.calls[0]["questions"]


def test_unknown_not_deferred_when_conf_at_least_080():
    env, mask = _map_env(
        [("UNKNOWN", (0, 1)), ("MONSTER", (1, 1))],
        hp=20,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 2.5},
                "pick": {"choice": "map_0", "confidence": 0.80},
            }
        ]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] != UNKNOWN_DEFERRED_REASON
    assert action == _MAP_START


def test_unknown_not_deferred_when_hp_pressure_below_2():
    env, mask = _map_env(
        [("UNKNOWN", (0, 1)), ("MONSTER", (1, 1))],
        hp=70,
        max_hp=80,
    )
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 1.4},
                "pick": {"choice": "map_0", "confidence": 0.70},
            }
        ]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _MAP_START
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] != UNKNOWN_DEFERRED_REASON


def test_unknown_not_deferred_when_no_non_unknown_legal():
    env, mask = _map_env([("UNKNOWN", (0, 1))], hp=10, max_hp=80)
    adapter = ScriptedJev(
        [
            {
                "hp_pressure": {"score": 3.0},
                "pick": {"choice": "map_0", "confidence": 0.70},
            }
        ]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _MAP_START
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] != UNKNOWN_DEFERRED_REASON


def test_build_event_options_maps_event_choice_to_event_start():
    actions = [
        {
            "action": "event_choice",
            "option_id": "TAKE_GOLD",
            "label": "Take gold",
            "description": "+50 gold",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "LEAVE",
            "label": "Leave",
            "description": "nothing",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "HIDDEN",
            "label": "Hidden",
            "description": "no",
            "enabled": False,
        },
    ]
    env = _event_env(actions)
    mask = _event_mask(2)
    cands = collect_candidates(env, mask)
    assert [c.key for c in cands] == ["TAKE_GOLD", "LEAVE"]
    assert cands[0].run_action == _EVENT_START
    assert cands[1].run_action == _EVENT_START + 1
    assert "Take gold: +50 gold" in cands[0].description
    assert all(c.payload.get("list_index") in {0, 1} for c in cands)
    built = build_event_options(actions, lambda idx: True, event_id="BrainLeech")
    assert [c.key for c in built] == ["TAKE_GOLD", "LEAVE"]


def test_event_pending_choose_confirm_maps_to_combat_slots():
    actions = [
        {"action": "confirm_choice", "prompt": "done"},
        {"action": "choose", "index": 0, "card_id": "STRIKE", "label": "Strike"},
        {"action": "choose", "index": 1, "card_id": "BASH", "label": "Bash"},
    ]
    env = _event_env(actions, event_id="BrainLeech")
    env._mgr._event_model = SimpleNamespace(event_id="BrainLeech", pending_choice=object())
    mask = _event_mask(pending=True, n_choose=2)
    cands = collect_candidates(env, mask)
    assert cands[0].run_action == _COMBAT_START
    assert cands[1].run_action == _COMBAT_START + 1
    assert cands[2].run_action == _COMBAT_START + 2
    adapter = ScriptedJev(
        [{CHOICE_EVENT: {"choice": cands[1].key, "confidence": 0.9}}]
    )
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls, "pending EVENT choose/confirm must call Jev, not fail-open random"
    assert CHOICE_EVENT in adapter.calls[0]["questions"]
    assert action == _COMBAT_START + 1
    assert log["jev_choice_id"] == CHOICE_EVENT
    assert log["phase"] == "EVENT"


def test_jev_event_off_does_not_call_adapter():
    actions = [
        {
            "action": "event_choice",
            "option_id": "A",
            "label": "A",
            "description": "gold",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "B",
            "label": "B",
            "description": "leave",
            "enabled": True,
        },
    ]
    env = _event_env(actions)
    mask = _event_mask(2)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "A", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == JEV_EVENT_OFF_REASON
    assert mask[action] == 1
    assert log["phase"] == "EVENT"


def test_jev_event_on_uses_event_choice_name():
    actions = [
        {
            "action": "event_choice",
            "option_id": "A",
            "label": "Take gold",
            "description": "+50 gold (待核)",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "B",
            "label": "Take relic",
            "description": "a relic",
            "enabled": True,
        },
    ]
    env = _event_env(actions, event_id="TeaMaster")
    mask = _event_mask(2)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "A", "confidence": 0.9}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert action == _EVENT_START
    q = adapter.calls[0]["questions"]
    assert CHOICE_EVENT in q
    assert CHOICE_NEOW_BOON not in q
    assert "待核" in q[CHOICE_EVENT]["instructions"]
    assert "content_map" in adapter.calls[0]["state"]
    assert adapter.calls[0]["state"]["event_id"] == "TeaMaster"
    assert log["jev_choice_id"] == CHOICE_EVENT
    assert log["shadow_decision"] == "event_choice"
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_neow_boon_choice_name_when_detected():
    actions = [
        {
            "action": "event_choice",
            "option_id": "MAX_HP",
            "label": "Max HP",
            "description": "+8 max HP",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "GOLD",
            "label": "Gold",
            "description": "+100 gold",
            "enabled": True,
        },
    ]
    env = _event_env(actions, event_id="Neow")
    mask = _event_mask(2)
    adapter = ScriptedJev([{CHOICE_NEOW_BOON: {"choice": "GOLD", "confidence": 0.91}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert CHOICE_NEOW_BOON in adapter.calls[0]["questions"]
    assert CHOICE_EVENT not in adapter.calls[0]["questions"]
    assert "mid" in adapter.calls[0]["questions"][CHOICE_NEOW_BOON]["instructions"].lower()
    assert action == _EVENT_START + 1
    assert log["jev_choice_id"] == CHOICE_NEOW_BOON
    assert log["is_neow"] is True
    assert log["phase"] == "NEOW"


def test_neow_off_skips_jev_silently():
    actions = [
        {
            "action": "event_choice",
            "option_id": "MAX_HP",
            "label": "Max HP",
            "description": "+8",
            "enabled": True,
        }
    ]
    env = _event_env(actions, event_id="Neow")
    mask = _event_mask(1)
    adapter = ScriptedJev([{CHOICE_NEOW_BOON: {"choice": "MAX_HP", "confidence": 0.99}}])
    flags = JevPolicyFlags(
        phases=frozenset({"map", "rest", "card", "event"}),
        event=True,
        neow=False,
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter, flags=flags)
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == JEV_NEOW_OFF_REASON
    assert log["is_neow"] is True
    assert mask[action] == 1


def test_neow_leave_only_does_not_call_jev():
    actions = [
        {
            "action": "event_choice",
            "option_id": "leave",
            "label": "Leave",
            "description": "Leave",
            "enabled": True,
        }
    ]
    env = _event_env(actions, event_id="Neow")
    mask = _event_mask(1)
    adapter = ScriptedJev([{CHOICE_NEOW_BOON: {"choice": "leave", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == NEOW_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert log["is_neow"] is True
    assert mask[action] == 1
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_neow_one_boon_does_not_call_jev():
    actions = [
        {
            "action": "event_choice",
            "option_id": "leave",
            "label": "Leave",
            "enabled": True,
        },
        {
            "action": "event_choice",
            "option_id": "MAX_HP",
            "label": "Max HP",
            "enabled": True,
        },
    ]
    env = _event_env(actions, event_id="Neow")
    mask = _event_mask(2)
    adapter = ScriptedJev([{CHOICE_NEOW_BOON: {"choice": "MAX_HP", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == NEOW_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert mask[action] == 1


def test_neow_empty_options_does_not_call_jev():
    env = _event_env([], event_id="Neow")
    mask = _event_mask(1)
    adapter = ScriptedJev([{CHOICE_NEOW_BOON: {"choice": "leave", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == NEOW_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert log["is_neow"] is True
    assert mask[action] == 1


def _event_choice(option_id, label, description=""):
    return {
        "action": "event_choice",
        "option_id": option_id,
        "label": label,
        "description": description,
        "enabled": True,
    }


def test_event_leave_only_does_not_call_jev():
    actions = [_event_choice("leave", "Leave", "Leave")]
    env = _event_env(actions, event_id="BrainLeech")
    mask = _event_mask(1)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "leave", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == EVENT_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert log["is_neow"] is False
    assert log["phase"] == "EVENT"
    assert mask[action] == 1


def test_event_one_non_leave_does_not_call_jev():
    actions = [
        _event_choice("leave", "Leave"),
        _event_choice("GOLD", "Take gold", "+50 gold"),
    ]
    env = _event_env(actions)
    mask = _event_mask(2)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "GOLD", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == EVENT_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert mask[action] == 1


def test_event_empty_options_does_not_call_jev():
    env = _event_env([])
    mask = _event_mask(1)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "A", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == EVENT_OPTIONS_EMPTY_REASON
    assert log["shadow_status"] == "skipped"
    assert log["shadow_decision"] == "event_choice"
    assert mask[action] == 1


def test_event_two_non_leave_still_calls_jev():
    actions = [
        _event_choice("GOLD", "Take gold", "+50"),
        _event_choice("RELIC", "Take relic", "a relic"),
        _event_choice("leave", "Leave", "nothing"),
    ]
    env = _event_env(actions, event_id="TeaMaster")
    mask = _event_mask(3)
    adapter = ScriptedJev([{CHOICE_EVENT: {"choice": "GOLD", "confidence": 0.9}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls, "≥2 non-Leave EVENT options must call Choice"
    assert CHOICE_EVENT in adapter.calls[0]["questions"]
    assert action == _EVENT_START
    assert log["jev_choice_id"] == CHOICE_EVENT
    assert log["shadow_status"] == "ok"


def test_shop_does_not_call_jev():
    actions = [
        {"action": "leave_shop"},
        {"action": "buy_card", "card_id": "ANGER"},
    ]
    env = _env(RunManager.PHASE_SHOP, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_SHOP_START] = 1
    mask[_SHOP_START + 1] = 1
    adapter = ScriptedJev([{"pick": {"choice": "shop_leave", "confidence": 0.99}}])
    action, log = choose_jev_noncombat(
        env, mask, np.random.RandomState(0), adapter, flags=EVENT_ON
    )
    assert adapter.calls == []
    assert log["shadow_fallback_reason"] == SHOP_RANDOM_REASON
    assert mask[action] == 1


def test_resolve_jev_flags_default_event_off():
    flags = resolve_jev_flags()
    assert flags.allows_event() is False
    assert flags.allows_neow() is False
    assert "event" not in flags.resolved_phases()
    on = resolve_jev_flags(jev_event="on")
    assert on.allows_event() is True
    assert on.allows_neow() is True
    via_phases = resolve_jev_flags(jev_phases="map,rest,card,event")
    assert via_phases.allows_event() is True
    neow_off = resolve_jev_flags(jev_event="on", jev_neow="off")
    assert neow_off.allows_event() is True
    assert neow_off.allows_neow() is False


def _rest_pending_mask(n_choose, *, confirm=False):
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    if confirm:
        mask[_COMBAT_START] = 1
    for i in range(n_choose):
        mask[_COMBAT_START + 1 + i] = 1
    return mask


def test_build_options_rest_site_maps_pending_choose_to_combat_slots():
    n = 14
    actions = [{"action": "confirm_choice", "prompt": "Choose a card to upgrade"}] + [
        {"action": "choose", "index": i, "card_id": f"CARD_{i}"}
        for i in range(n)
    ]
    env = _env(RunManager.PHASE_REST_SITE, actions)
    mask = _rest_pending_mask(n, confirm=True)
    cands = build_options(
        actions, lambda idx: int(mask[idx]) == 1, phase=RunManager.PHASE_REST_SITE
    )
    keys = [c.key for c in cands]
    assert keys[0] == "rest_confirm"
    assert keys[1:] == [f"rest_choose_{i}" for i in range(n)]
    assert len([k for k in keys if k.startswith("rest_choose_")]) == n
    assert cands[0].run_action == _COMBAT_START
    assert cands[1].run_action == _COMBAT_START + 1
    assert cands[-1].run_action == _COMBAT_START + n
    assert all(c.legal for c in cands)
    kept = strip_illegal_invisible(collect_candidates(env, mask))
    assert [c.key for c in kept] == keys
    adapter = ScriptedJev([{"pick": {"choice": "rest_choose_3", "confidence": 0.9}}])
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert adapter.calls, "REST pending must call Jev, not empty legal_ids"
    assert action == _COMBAT_START + 1 + 3
    assert log["shadow_status"] == "ok"
    assert log["shadow_decision"] == "rest_site"
    assert log["legal_ids"] == keys
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_rest_smith_pending_live_rest_choose_keys():
    env = STS2RunEnv(character_id="Ironclad")
    env.reset(seed=200008)
    env._mgr._enter_rest_site()
    smith = next(
        a for a in env._mgr.get_available_actions() if a.get("option_id") == "SMITH"
    )
    env._mgr.take_action(smith)
    mask = env.action_masks()
    kept = strip_illegal_invisible(collect_candidates(env, mask))
    assert kept, "Smith pending must not yield empty legal_ids"
    assert all(c.key.startswith("rest_choose_") for c in kept)
    assert all(mask[c.run_action] == 1 for c in kept)
    assert len(kept) >= 10
    env.close()


def test_build_options_rest_site_still_maps_rest_option():
    actions = [
        {
            "action": "rest_option",
            "option_id": "HEAL",
            "label": "Rest",
            "enabled": True,
        },
        {
            "action": "rest_option",
            "option_id": "SMITH",
            "label": "Smith",
            "enabled": True,
        },
    ]
    env = _env(RunManager.PHASE_REST_SITE, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_REST_START] = 1
    mask[_REST_START + 1] = 1
    cands = build_rest_options(actions, lambda idx: int(mask[idx]) == 1)
    assert [c.key for c in cands] == ["rest_HEAL", "rest_SMITH"]
    assert cands[0].run_action == _REST_START
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 1.5}, "pick": {"choice": "rest_SMITH", "confidence": 0.9}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _REST_START + 1
    assert log["shadow_status"] == "ok"
    assert CHOICE_CONFIDENCE_MIN == 0.65
    assert REST_CHOICE_MIN_CONFIDENCE == 0.50


def _rest_heal_smith_env():
    actions = [
        {
            "action": "rest_option",
            "option_id": "HEAL",
            "label": "Rest",
            "enabled": True,
        },
        {
            "action": "rest_option",
            "option_id": "SMITH",
            "label": "Smith",
            "enabled": True,
        },
    ]
    env = _env(RunManager.PHASE_REST_SITE, actions)
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    mask[_REST_START] = 1
    mask[_REST_START + 1] = 1
    return env, mask


def test_rest_soft_min_050_lands_and_049_does_not():
    env, mask = _rest_heal_smith_env()
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 1.5}, "pick": {"choice": "rest_SMITH", "confidence": 0.50}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _REST_START + 1
    assert log["shadow_status"] == "ok"
    assert REST_CHOICE_MIN_CONFIDENCE == 0.50
    env, mask = _rest_heal_smith_env()
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 1.5}, "pick": {"choice": "rest_SMITH", "confidence": 0.49}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert log["shadow_status"] == "uncertain"
    assert mask[action] == 1
    assert CHOICE_CONFIDENCE_MIN == 0.65


def test_rest_heal_assist_after_hp_pressure_bias():
    env, mask = _rest_heal_smith_env()
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 2.5}, "pick": {"choice": "rest_HEAL", "confidence": 0.30}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _REST_START
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] == HP_PRESSURE_ASSIST_REASON
    assert REST_HEAL_ASSIST_CONF == 0.30


def test_rest_heal_assist_requires_conf_030():
    env, mask = _rest_heal_smith_env()
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 2.5}, "pick": {"choice": "rest_HEAL", "confidence": 0.29}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert log["shadow_status"] == "uncertain"
    assert log["shadow_fallback_reason"] != HP_PRESSURE_ASSIST_REASON
    assert mask[action] == 1


def test_rest_smith_assist_after_hp_pressure_bias():
    env, mask = _rest_heal_smith_env()
    adapter = ScriptedJev(
        [{"hp_pressure": {"score": 0.5}, "pick": {"choice": "rest_SMITH", "confidence": 0.40}}]
    )
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert action == _REST_START + 1
    assert log["shadow_status"] == "ok"
    assert log["shadow_fallback_reason"] == SMITH_ASSIST_REASON
    assert REST_SMITH_ASSIST_CONF == 0.40


def test_map_choice_threshold_stays_065():
    env, mask = _map_env([("MONSTER", (0, 1)), ("ELITE", (1, 1))])
    adapter = ScriptedJev([{"pick": {"choice": "map_1", "confidence": 0.64}}])
    action, log = choose_jev_noncombat(env, mask, np.random.RandomState(0), adapter)
    assert log["shadow_status"] == "uncertain"
    assert CHOICE_CONFIDENCE_MIN == 0.65
    assert mask[action] == 1
