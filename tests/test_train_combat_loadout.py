"""Combat train loadout: neow_early rotating fixtures (no training)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.core.enums import CardId
from sts2_env.gym_env.combat_env import STS2CombatEnv

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_combat.py"
_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "neow_early"


def _load_train_mod():
    spec = importlib.util.spec_from_file_location("train_combat", _TRAIN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_mod = _load_train_mod()


def test_cli_loadout_neow_early_parses():
    args = train_mod.parse_args(["--loadout", "neow_early"])
    assert args.loadout == "neow_early"
    bare = train_mod.parse_args([])
    assert bare.loadout == "bare"


def test_cli_loadout_mix_neow_v1_aliases():
    assert train_mod.LOADOUT_SUITES["mix_neow_v1"] == {"mix": ("neow_early", "loadout_v1")}
    for alias in ("mix", "mix_neow", "mid_early", "early_mid", "mix_neow_v1"):
        args = train_mod.parse_args(["--loadout", alias])
        assert args.loadout == "mix_neow_v1"


def test_neow_early_fixtures_locked_and_labelled():
    fixtures = train_mod.load_neow_early_fixtures(_FIXTURE_DIR)
    assert len(fixtures) == 50
    train_mod.assert_neow_early_labels(fixtures)
    paths = list(_FIXTURE_DIR.glob("neow_early_*.json"))
    assert len(paths) == 50
    for fx in fixtures:
        assert fx.get("locked") is True
        label = str(fx["label"])
        assert "Neow+early" in label
        assert label.strip().lower() != "bare"
        assert not label.lower().startswith("bare")
        assert fx.get("deck")
        assert int(fx["max_hp"]) > 0
        assert int(fx["hp"]) > 0
        if int(fx.get("snapshot_hp", 0)) <= 0:
            expected, synth = train_mod.synth_entry_hp(0, int(fx["max_hp"]))
            assert synth
            assert int(fx["hp"]) == expected


def test_synth_entry_hp_from_death_snapshot():
    hp, synth = train_mod.synth_entry_hp(0, 80)
    assert synth
    assert hp == round(0.55 * 80)
    hp2, synth2 = train_mod.synth_entry_hp(44, 80)
    assert not synth2
    assert hp2 == 44


def test_combat_env_rotates_neow_early_not_bare_starter():
    fixtures = train_mod.load_neow_early_fixtures(_FIXTURE_DIR)
    provider = train_mod.RotatingNeowEarlyProvider(fixtures, offset=0)
    env = STS2CombatEnv(loadout_provider=provider)
    env.reset(seed=0)
    assert env.combat is not None
    combat = env.combat
    cards = list(combat.draw_pile) + list(combat.hand) + list(combat.discard_pile) + list(
        combat.exhaust_pile
    )
    fx0 = fixtures[0]
    assert combat.player.max_hp == int(fx0["max_hp"])
    assert combat.player.current_hp == int(fx0["hp"])
    assert combat.player.current_hp > 0
    assert len(cards) == len(fx0["deck"])
    env.close()


def test_make_loadout_provider_bare_is_none():
    assert train_mod.make_loadout_provider("bare") is None
    provider = train_mod.make_loadout_provider("neow_early", fixture_dir=_FIXTURE_DIR)
    spec = provider()
    assert spec["hp"] > 0
    assert spec["deck"]
    assert spec["deck"][0].card_id in CardId
    assert IRONCLAD_STARTING_HP == 80


def test_mix_neow_v1_interleaves_fifty_fifty():
    provider = train_mod.make_loadout_provider("mix_neow_v1", offset=0, fixture_dir=_FIXTURE_DIR)
    suites = [provider()["suite"] for _ in range(20)]
    assert suites[0::2] == ["neow_early"] * 10
    assert suites[1::2] == ["loadout_v1"] * 10
    v1 = train_mod.load_loadout_v1_fixtures()
    assert len(v1) == 50
    assert all("loadout_v1" in str(fx["label"]) for fx in v1)
    assert all(not str(fx["label"]).startswith("Neow+early") for fx in v1)


def test_loadout_v1_materializes():
    provider = train_mod.make_loadout_provider("loadout_v1")
    spec = provider()
    assert spec["suite"] == "loadout_v1"
    assert spec["hp"] > 0
    assert spec["deck"]
    env = STS2CombatEnv(loadout_provider=provider)
    env.reset(seed=1)
    assert env.combat is not None
    assert env.combat.player.current_hp > 0
    env.close()


def test_jev_noncombat_script_surface():
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "jev_noncombat.py"
    spec = importlib.util.spec_from_file_location("jev_noncombat", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.POTION_OR_RELIC_REASON == "potion_or_relic_reward_random"
    assert mod.CARD_FIT_ASSIST_REASON == "jev_card_fit_assist"
    assert mod.CHOICE_CONFIDENCE_MIN == 0.65
    assert mod.REST_CHOICE_MIN_CONFIDENCE == 0.50
    assert "Neow+early" in mod.NEOW_EARLY_CARD_INSTRUCTIONS
    assert mod.NEOW_JEV_OFF_REASON == "neow_jev_off_random"
    assert "jev_neow" in mod.decide_noncombat.__code__.co_varnames
    assert mod.is_potion_or_relic_reward([{"action": "pick_potion"}])
