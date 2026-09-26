"""Death-attribution census: summarize_traces, opt-in _run_episode, CLI flag."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sts2_env.eval.act1_runner import _run_episode
from sts2_env.eval.death_census import (
    CombatEvent,
    RunTrace,
    ShopVisit,
    summarize_traces,
)
from sts2_env.gym_env.run_env import STS2RunEnv

_EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_act1_runenv.py"


def _load_eval_mod():
    spec = importlib.util.spec_from_file_location("eval_act1_runenv", _EVAL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_mod = _load_eval_mod()


def _base_trace(**overrides) -> RunTrace:
    defaults = dict(
        seed=0,
        act1_clear=False,
        truncated=False,
        death_kind="noncombat",
        death_floor=0,
        death_hp=0,
        death_gold=0,
        death_room_kind="map",
        death_encounter_id="",
        death_enemy_ids=[],
        combats=[],
        shop_visits=[],
        rest_visits=0,
        event_visits=0,
        card_picks=0,
        card_skips=0,
        deck_size=10,
        upgraded_cards=0,
        relic_count=0,
        potion_count=0,
    )
    defaults.update(overrides)
    return RunTrace(**defaults)


def test_summarize_traces_literal_counts():
    elite_lost = CombatEvent(
        floor=10,
        act=0,
        room_kind="elite",
        encounter_id="e1",
        enemy_ids=["slime"],
        hp_in=50,
        hp_out=0,
        max_hp=80,
        gold_in=99,
        gold_out=40,
        outcome="lost",
        steps=12,
    )
    boss_lost = CombatEvent(
        floor=50,
        act=0,
        room_kind="boss",
        encounter_id="b1",
        enemy_ids=["boss"],
        hp_in=30,
        hp_out=0,
        max_hp=80,
        gold_in=80,
        gold_out=80,
        outcome="lost",
        steps=20,
    )
    monster_lost = CombatEvent(
        floor=8,
        act=0,
        room_kind="monster",
        encounter_id="m1",
        enemy_ids=["jaw"],
        hp_in=60,
        hp_out=0,
        max_hp=80,
        gold_in=120,
        gold_out=120,
        outcome="lost",
        steps=8,
    )
    traces = [
        _base_trace(
            seed=1,
            death_kind="combat_elite",
            death_floor=10,
            death_hp=0,
            death_gold=40,
            death_room_kind="elite",
            combats=[elite_lost],
            shop_visits=[ShopVisit(floor=5, gold_in=99, gold_out=40, spent=59)],
        ),
        _base_trace(
            seed=2,
            death_kind="combat_boss",
            death_floor=50,
            death_hp=0,
            death_gold=80,
            death_room_kind="boss",
            combats=[boss_lost],
        ),
        _base_trace(
            seed=3,
            death_kind="combat_monster",
            death_floor=8,
            death_hp=0,
            death_gold=120,
            death_room_kind="monster",
            combats=[monster_lost],
        ),
        _base_trace(
            seed=4,
            act1_clear=True,
            death_kind="cleared",
            death_floor=17,
            death_hp=45,
            death_gold=200,
            death_room_kind="map",
            combats=[
                CombatEvent(
                    floor=6,
                    act=0,
                    room_kind="monster",
                    encounter_id="m0",
                    enemy_ids=["x"],
                    hp_in=70,
                    hp_out=55,
                    max_hp=80,
                    gold_in=99,
                    gold_out=100,
                    outcome="won",
                    steps=5,
                )
            ],
            upgraded_cards=2,
            deck_size=15,
            relic_count=3,
        ),
    ]
    s = summarize_traces(traces)
    assert s["n"] == 4
    assert s["act1_clear_rate"] == 0.25
    assert s["trunc_rate"] == 0.0
    assert s["death_kind_counts"] == {
        "combat_monster": 1,
        "combat_elite": 1,
        "combat_boss": 1,
        "noncombat": 0,
        "truncation": 0,
        "cleared": 1,
    }
    assert s["n_deaths_with_unspent_gold_ge_99"] == 1
    assert s["combats_lost_by_room"] == {"monster": 1, "elite": 1, "boss": 1}
    assert s["mean_death_floor"] == pytest.approx((10 + 50 + 8) / 3)
    assert s["mean_death_gold"] == pytest.approx((40 + 80 + 120) / 3)
    assert s["mean_gold_spent_in_shops_all_runs"] == pytest.approx(59 / 4)
    assert s["mean_hp_lost_in_won_combats_by_room"]["monster"] == pytest.approx(15.0)
    assert s["mean_upgraded_cards_at_clear"] == pytest.approx(2.0)
    assert s["mean_deck_size_at_clear"] == pytest.approx(15.0)
    assert s["mean_relic_count_at_clear"] == pytest.approx(3.0)


def test_run_episode_default_row_has_no_census_key():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=2000)
    rng = np.random.RandomState(0)
    try:
        row = _run_episode(
            env,
            "random",
            None,
            None,
            200000,
            rng,
            death_census=False,
        )
    finally:
        env.close()
    assert "death_census" not in row
    assert row["seed"] == 200000
    assert "act1_clear" in row
    assert "floor" in row
    assert "hp" in row
    assert "gold" in row


def test_run_episode_census_opt_in_has_required_fields():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=2000)
    rng = np.random.RandomState(0)
    try:
        row = _run_episode(
            env,
            "random",
            None,
            None,
            200000,
            rng,
            death_census=True,
        )
    finally:
        env.close()
    assert "death_census" in row
    trace = row["death_census"]
    for key in (
        "seed",
        "death_kind",
        "death_gold",
        "combats",
        "shop_visits",
        "upgraded_cards",
        "deck_size",
    ):
        assert key in trace
    if int(row["hp"]) == 0:
        dk = trace["death_kind"]
        assert dk.startswith("combat_") or dk == "noncombat"
    combats = trace["combats"]
    if combats and str(trace["death_kind"]).startswith("combat_"):
        assert combats[-1]["outcome"] == "lost"
        for fight in combats[:-1]:
            assert fight["outcome"] == "won"


def test_eval_cli_flag_default_off():
    args = eval_mod.parse_args([])
    assert args.death_census is False
    args_on = eval_mod.parse_args(["--death-census"])
    assert args_on.death_census is True
