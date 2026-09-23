"""Combat train loadout: neow_early rotating fixtures (no training)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
    tags = [str(fx.get("label") or fx.get("id") or "") for fx in v1]
    assert all("loadout_v1" in tag for tag in tags)
    assert all(not tag.startswith("Neow+early") for tag in tags)


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


def test_materialize_hold_fixtures_and_id_key_upgraded():
    from sts2_env.core.enums import CardId

    root = Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "loadout_v1"
    for stem in ("loadout_v1_01", "loadout_v1_02", "loadout_v1_03"):
        fx = train_mod.load_json_fixtures([root / f"{stem}.json"])[0]
        spec = train_mod.materialize_fixture(fx, suite="loadout_v1")
        assert spec["suite"] == "loadout_v1"
        assert spec["deck"]
        assert spec["hp"] > 0
        assert all(c.card_id in CardId for c in spec["deck"])
        assert spec["relics"]
        assert spec["potions"]
    spec = train_mod.materialize_fixture(
        {
            "hp": 50,
            "max_hp": 80,
            "deck": [
                {"id": "BASH", "upgraded": True},
                {"card_id": "STRIKE_IRONCLAD", "upgraded": False},
                "DEFEND_IRONCLAD",
            ],
        },
        suite="loadout_v1",
    )
    assert spec["deck"][0].card_id == CardId.BASH
    assert spec["deck"][0].upgraded is True
    assert spec["deck"][1].card_id == CardId.STRIKE_IRONCLAD
    assert spec["deck"][1].upgraded is False
    assert spec["deck"][2].card_id == CardId.DEFEND_IRONCLAD
    assert "relics" not in spec
    with pytest.raises(SystemExit, match="null entry"):
        train_mod.materialize_fixture({"hp": 50, "max_hp": 80, "deck": [None]})
    with pytest.raises(SystemExit, match="unknown card_id"):
        train_mod.materialize_fixture(
            {"hp": 50, "max_hp": 80, "deck": [{"upgraded": True}]}
        )


def test_hang_hold_fixtures_keep_distinct_relics_and_potions():
    """Hang-era 01/02/03 must not be homogenized to Shuriken-only."""
    from sts2_env.eval.combat_hold import apply_hold_relics, options_from_hold_fixture
    from sts2_env.relics.base import RelicId

    root = Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "loadout_v1"
    fx01, fx02, fx03 = [
        train_mod.load_json_fixtures([root / f"{stem}.json"])[0]
        for stem in ("loadout_v1_01", "loadout_v1_02", "loadout_v1_03")
    ]
    assert fx01["hp"] == 58
    assert fx01["relics"] == ["BURNING_BLOOD", "SHURIKEN"]
    assert fx01["potions"] == ["FirePotion", "AttackPotion", None]
    assert fx02["relics"] == ["BURNING_BLOOD", "BAG_OF_MARBLES"]
    assert fx02["potions"] == ["ExplosiveAmpoule", "BlockPotion", None]
    assert fx03["relics"] == ["BURNING_BLOOD", "VAJRA"]
    assert fx03["potions"] == ["StrengthPotion", "FlexPotion", None]

    spec02 = train_mod.materialize_fixture(fx02, suite="loadout_v1")
    assert spec02["relics"] == ["BURNING_BLOOD", "BAG_OF_MARBLES"]
    assert apply_hold_relics(dict(spec02), fixture=fx02)["relics"] == [
        "BURNING_BLOOD",
        "BAG_OF_MARBLES",
    ]
    spec01 = train_mod.materialize_fixture(fx01, suite="loadout_v1")
    assert [p.potion_id if p else None for p in spec01["potions"]] == [
        "FirePotion",
        "AttackPotion",
        None,
    ]
    spec03 = train_mod.materialize_fixture(fx03, suite="loadout_v1")
    assert "VAJRA" in spec03["relics"]

    env = STS2CombatEnv()
    env.reset(seed=40000, options=options_from_hold_fixture(fx01))
    assert env.combat is not None
    ids01 = {r.relic_id for r in env.combat.relics}
    pots01 = [p.potion_id if p else None for p in env.combat.potions]
    assert RelicId.SHURIKEN in ids01
    assert "FirePotion" in pots01
    env.close()

    env = STS2CombatEnv()
    env.reset(seed=41000, options=options_from_hold_fixture(fx02))
    assert env.combat is not None
    ids02 = {r.relic_id for r in env.combat.relics}
    pots02 = [p.potion_id if p else None for p in env.combat.potions]
    assert RelicId.BAG_OF_MARBLES in ids02
    assert RelicId.SHURIKEN not in ids02
    assert "ExplosiveAmpoule" in pots02
    env.close()

    env = STS2CombatEnv()
    env.reset(seed=42000, options=options_from_hold_fixture(fx03))
    assert env.combat is not None
    ids03 = {r.relic_id for r in env.combat.relics}
    pots03 = [p.potion_id if p else None for p in env.combat.potions]
    assert RelicId.VAJRA in ids03
    assert "StrengthPotion" in pots03
    env.close()


def test_materialize_keeps_relics_and_potions_env_applies():
    from sts2_env.eval.combat_hold import HOLD_DEFAULT_RELICS, apply_hold_relics
    from sts2_env.relics.base import RelicId

    spec = train_mod.materialize_fixture(
        {
            "hp": 50,
            "max_hp": 80,
            "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
            "relics": ["BURNING_BLOOD", "Shuriken"],
            "potions": ["FirePotion", {"id": "StrengthPotion"}],
        },
        suite="loadout_v1",
    )
    assert spec["relics"] == ["BURNING_BLOOD", "SHURIKEN"]
    assert [p.potion_id for p in spec["potions"]] == ["FirePotion", "StrengthPotion"]
    opts = train_mod.options_from_fixture(
        {
            "hp": 50,
            "max_hp": 80,
            "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
            "relics": ["BURNING_BLOOD", "SHURIKEN"],
            "potions": ["FirePotion"],
        },
        suite="loadout_v1",
    )
    assert opts["relics"] == ["BURNING_BLOOD", "SHURIKEN"]
    env = STS2CombatEnv(loadout_provider=lambda: spec)
    env.reset(seed=7)
    assert env.combat is not None
    relic_ids = {r.relic_id for r in env.combat.relics}
    assert RelicId.BURNING_BLOOD in relic_ids
    assert RelicId.SHURIKEN in relic_ids
    potion_ids = [p.potion_id for p in env.combat.potions if p is not None]
    assert "FirePotion" in potion_ids
    assert "StrengthPotion" in potion_ids
    env.close()

    env2 = STS2CombatEnv()
    env2.reset(seed=8, options=opts)
    assert env2.combat is not None
    relic_ids2 = {r.relic_id for r in env2.combat.relics}
    assert RelicId.BURNING_BLOOD in relic_ids2
    assert RelicId.SHURIKEN in relic_ids2
    env2.close()

    omitted = train_mod.materialize_fixture(
        {"hp": 50, "max_hp": 80, "deck": ["STRIKE_IRONCLAD", "BASH"]},
        suite="loadout_v1",
    )
    assert "relics" not in omitted
    assert apply_hold_relics(dict(omitted))["relics"] == list(HOLD_DEFAULT_RELICS)
    locked_empty = {"hp": 50, "max_hp": 80, "deck": ["BASH"], "relics": []}
    empty_spec = train_mod.materialize_fixture(locked_empty, suite="loadout_v1")
    assert empty_spec["relics"] == []
    assert apply_hold_relics(dict(empty_spec), fixture=locked_empty)["relics"] == []
    with pytest.raises(SystemExit, match="unknown relic"):
        train_mod.materialize_fixture(
            {
                "hp": 50,
                "max_hp": 80,
                "deck": ["BASH"],
                "relics": ["NOT_A_RELIC"],
            }
        )


def test_eval_combat_suite_cli_is_hang_hold():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_combat_suite.py"
    spec = importlib.util.spec_from_file_location("eval_combat_suite", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    args = mod.parse_args([])
    assert args.suite == "loadout_v1"
    assert args.n_eps == 20
    assert args.combat_policy == "ppo"
    assert args.workers == 1
    args20 = mod.parse_args(["--suite", "loadout_v1", "--n-eps", "20"])
    assert args20.n_eps == 20
    args8 = mod.parse_args(["--workers", "8", "--combat-policy", "jev"])
    assert args8.workers == 8
    assert args8.combat_policy == "jev"
    with pytest.raises(SystemExit, match="loadout_v1"):
        mod.main(["--suite", "bare"])


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
    assert "map_lowhp" in mod.decide_noncombat.__code__.co_varnames
    assert mod.MAP_LOWHP_HARD_REASON == "map_lowhp_hard"
    assert mod.MAP_LOWHP_SAFE_REASON == "map_lowhp_safe"
    assert mod.MAP_LOWHP_RANDOM_REASON == "map_lowhp_random"
    assert mod.MAP_LOWHP_ON is True
    assert mod.is_potion_or_relic_reward([{"action": "pick_potion"}])
    assert mod.TYPESAFE_HTTP_USER_AGENT == "sts2-rl-agent-jev/1.0"
    assert "Python-urllib" not in mod.TYPESAFE_HTTP_USER_AGENT
