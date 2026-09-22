"""Tests for frozen Act1 RunEnv eval: obs_v1 sizes, CLI, hierarchical policy."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sts2_env.core.enums import IntentType
from sts2_env.gym_env.observation import (
    ENEMY_FEATURES,
    INTENT_TYPES,
    NUM_INTENT_TYPES,
    OBS_SIZE,
    encode_observation,
)
from sts2_env.gym_env.run_env import (
    COMBAT_OBS_SIZE,
    RUN_OBS_SIZE,
    STS2RunEnv,
    _COMBAT_SIZE,
    _COMBAT_START,
)
from sts2_env.run.run_manager import RunManager

_EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_act1_runenv.py"


def _load_eval_mod():
    spec = importlib.util.spec_from_file_location("eval_act1_runenv", _EVAL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_mod = _load_eval_mod()


class FakeSpace:
    def __init__(self, dim: int):
        self.shape = (dim,)


class FakeCombatModel:
    """Stand-in for MaskablePPO when the hung zip is not on the CI VM."""

    def __init__(self):
        self.observation_space = FakeSpace(OBS_SIZE)
        self.seen_widths: list[int] = []
        self.seen_masks: list[int] = []

    def predict(self, obs, action_masks=None, deterministic=True):
        arr = np.asarray(obs)
        width = int(arr.shape[-1])
        self.seen_widths.append(width)
        assert width == OBS_SIZE, f"combat model got obs width {width}, expected {OBS_SIZE}"
        assert width != RUN_OBS_SIZE, "combat model must not receive RunEnv obs"
        mask = np.asarray(action_masks)
        assert mask.shape[-1] == _COMBAT_SIZE
        self.seen_masks.append(int(mask.shape[-1]))
        valid = np.flatnonzero(mask == 1)
        assert valid.size > 0
        return int(valid[0]), None


def test_obs_v1_full_intent_onehot_is_181():
    assert INTENT_TYPES == list(IntentType)
    assert NUM_INTENT_TYPES == len(list(IntentType))
    assert IntentType.STUN in INTENT_TYPES
    assert IntentType.STATUS_CARD in INTENT_TYPES
    assert IntentType.SUMMON in INTENT_TYPES
    assert OBS_SIZE == 4 + 6 + 50 + 6 + 5 * ENEMY_FEATURES
    assert OBS_SIZE == 181
    assert COMBAT_OBS_SIZE == OBS_SIZE
    assert RUN_OBS_SIZE == OBS_SIZE + 20
    assert RUN_OBS_SIZE == 201
    assert RUN_OBS_SIZE != OBS_SIZE


def test_encode_observation_width_matches_obs_v1():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=80)
    obs, info = env.reset(seed=42)
    assert obs.shape == (RUN_OBS_SIZE,)
    rng = np.random.RandomState(0)
    for _ in range(80):
        if info.get("phase") == RunManager.PHASE_COMBAT:
            combat = env._mgr.get_combat_state()
            assert combat is not None
            combat_obs = encode_observation(combat)
            assert combat_obs.shape == (OBS_SIZE,)
            env.close()
            return
        mask = info["action_mask"]
        valid = np.flatnonzero(mask == 1)
        obs, _, terminated, truncated, info = env.step(int(rng.choice(valid)))
        if terminated or truncated:
            break
    env.close()
    pytest.skip("No combat phase reached")


def test_cli_hierarchical_requires_combat_zip():
    args = eval_mod.parse_args(["--policy", "hierarchical"])
    with pytest.raises(SystemExit, match="--model"):
        eval_mod.validate_policy_args(args)


def test_cli_hierarchical_accepts_model_as_combat_zip():
    args = eval_mod.parse_args(
        [
            "--policy",
            "hierarchical",
            "--model",
            eval_mod.HUNG_COMBAT_ZIP,
        ]
    )
    eval_mod.validate_policy_args(args)
    assert args.combat_model == eval_mod.HUNG_COMBAT_ZIP
    assert args.jev == "off"
    assert eval_mod.OBS_SIZE == 181


def test_cli_model_rejects_combat_model_flag():
    args = eval_mod.parse_args(
        ["--policy", "model", "--model", "run.zip", "--combat-model", "combat.zip"]
    )
    with pytest.raises(SystemExit, match="hierarchical"):
        eval_mod.validate_policy_args(args)


def test_cli_hierarchical_rejects_mismatched_model_paths():
    args = eval_mod.parse_args(
        ["--policy", "hierarchical", "--combat-model", "combat.zip", "--model", "other.zip"]
    )
    with pytest.raises(SystemExit, match="same combat"):
        eval_mod.validate_policy_args(args)


def test_refuse_combat_zip_on_runenv_model_path():
    fake = SimpleNamespace(observation_space=FakeSpace(OBS_SIZE))
    with pytest.raises(SystemExit, match="RUN_OBS_SIZE|expected"):
        eval_mod.require_obs_dim(fake, RUN_OBS_SIZE, "RunEnv --model")


def test_refuse_runenv_zip_as_combat_model():
    fake = SimpleNamespace(observation_space=FakeSpace(RUN_OBS_SIZE))
    with pytest.raises(SystemExit, match="expected"):
        eval_mod.require_obs_dim(fake, OBS_SIZE, "hierarchical --combat-model")


def test_accept_matching_obs_dims():
    combat = SimpleNamespace(observation_space=FakeSpace(OBS_SIZE))
    run = SimpleNamespace(observation_space=FakeSpace(RUN_OBS_SIZE))
    assert eval_mod.require_obs_dim(combat, OBS_SIZE, "hierarchical --combat-model") == OBS_SIZE
    assert eval_mod.require_obs_dim(run, RUN_OBS_SIZE, "RunEnv --model") == RUN_OBS_SIZE


def test_hierarchical_maps_combat_index_into_run_layout():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=120)
    obs, info = env.reset(seed=42)
    rng = np.random.RandomState(1)
    model = FakeCombatModel()
    widths: list[int] = []
    saw_combat = False
    for _ in range(120):
        mask = info.get("action_mask")
        action, shadow = eval_mod.choose_action(
            "hierarchical",
            env,
            obs,
            info,
            mask,
            rng,
            model=None,
            combat_model=model,
            received_obs_widths=widths,
        )
        if info.get("phase") == RunManager.PHASE_COMBAT:
            saw_combat = True
            assert _COMBAT_START <= action < _COMBAT_START + _COMBAT_SIZE
            assert shadow["shadow_status"] == eval_mod.JEV_SHADOW_SKIPPED
            assert shadow["shadow_suggestion"] is None
        else:
            assert mask[action] == 1
            assert shadow["shadow_status"] == eval_mod.JEV_SHADOW_STUB
            assert shadow["shadow_suggestion"] is None
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    if not saw_combat:
        pytest.skip("No combat phase reached")
    assert model.seen_widths
    assert set(model.seen_widths) == {OBS_SIZE}
    assert set(widths) == {OBS_SIZE}
    assert RUN_OBS_SIZE not in model.seen_widths


def test_missing_zip_errors_before_sb3():
    with pytest.raises(SystemExit, match="not found"):
        eval_mod.load_maskable_ppo("/tmp/definitely-missing-combat-ppo.zip")


def test_write_report_summary_omits_rows(tmp_path):
    rows = [
        {
            "seed": 200000,
            "act1_clear": False,
            "full_run_win": False,
            "truncated": True,
            "max_act": 0,
            "floor": 3,
            "hp": 40,
            "max_hp": 80,
            "gold": 99,
            "steps": 10,
            "reward": -1.0,
        }
    ]
    report = eval_mod.build_report(
        policy="hierarchical",
        model_path="",
        combat_model_path="/tmp/combat.zip",
        rows=rows,
        elapsed_s=1.2,
    )
    out = tmp_path / "act1_runenv.json"
    summary_path = eval_mod.write_report(report, out)
    assert out.is_file()
    assert summary_path.name == "act1_runenv.summary.json"
    slim = json.loads(summary_path.read_text())
    assert "rows" not in slim
    assert slim["policy"] == "hierarchical"
    assert slim["combat_obs_size"] == OBS_SIZE
    assert slim["run_obs_size"] == RUN_OBS_SIZE
    assert slim["summary"]["n"] == 1
    assert slim["summary"]["trunc_rate"] == 1.0
    assert slim["jev_shadow"]["mode"] == "stub"


def test_jev_shadow_does_not_change_noncombat_action():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=20)
    obs, info = env.reset(seed=7)
    rng = np.random.RandomState(0)
    mask = info["action_mask"]
    if info.get("phase") == RunManager.PHASE_COMBAT:
        env.close()
        pytest.skip("seed opened in combat")
    action, shadow = eval_mod.choose_hierarchical_action(
        env, obs, mask, rng, FakeCombatModel()
    )
    env.close()
    assert mask[action] == 1
    assert shadow["shadow_status"] == eval_mod.JEV_SHADOW_STUB
    assert shadow["shadow_suggestion"] is None


def test_cli_jev_default_off_and_strategic_alias():
    off = eval_mod.parse_args(["--policy", "hierarchical", "--combat-model", "c.zip"])
    eval_mod.validate_policy_args(off)
    assert off.jev == "off"
    on = eval_mod.parse_args(
        ["--policy", "hierarchical", "--combat-model", "c.zip", "--jev", "on"]
    )
    eval_mod.validate_policy_args(on)
    assert on.jev == "on"
    alias = eval_mod.parse_args(
        ["--policy", "hierarchical", "--combat-model", "c.zip", "--strategic", "jev"]
    )
    eval_mod.validate_policy_args(alias)
    assert alias.jev == "on"


def test_cli_jev_on_requires_hierarchical():
    args = eval_mod.parse_args(["--policy", "random", "--jev", "on"])
    with pytest.raises(SystemExit, match="hierarchical"):
        eval_mod.validate_policy_args(args)


def test_write_report_jev_on_fields(tmp_path):
    rows = [
        {
            "seed": 200000,
            "act1_clear": False,
            "full_run_win": False,
            "truncated": True,
            "max_act": 0,
            "floor": 6,
            "hp": 40,
            "max_hp": 80,
            "gold": 99,
            "steps": 10,
            "reward": -1.0,
            "shadow_suggestion": "map_0",
            "shadow_status": "ok",
            "shadow_confidence": 0.8,
            "shadow_hp_pressure": 1.4,
            "shadow_fallback_reason": None,
        }
    ]
    report = eval_mod.build_report(
        policy="hierarchical",
        model_path="",
        combat_model_path="/tmp/combat.zip",
        rows=rows,
        elapsed_s=1.2,
        jev="on",
    )
    assert report["jev"] == "on"
    assert report["jev_shadow"]["mode"] == "on"
    out = tmp_path / "act1_runenv.json"
    summary_path = eval_mod.write_report(report, out)
    slim = json.loads(summary_path.read_text())
    assert slim["jev"] == "on"
    assert slim["combat_obs_size"] == OBS_SIZE


class _BoomIfCalled:
    def system_one(self, *args, **kwargs):
        raise AssertionError("Jev must not run on combat steps")


class _ErrorAdapter:
    def __init__(self):
        self.n = 0

    def system_one(self, state, questions):
        self.n += 1
        from sts2_env.eval.jev import JevError

        raise JevError("forced")


def test_jev_on_never_enters_combat_and_still_acts():
    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=120)
    obs, info = env.reset(seed=42)
    rng = np.random.RandomState(1)
    model = FakeCombatModel()
    adapter = _ErrorAdapter()
    saw_combat = False
    saw_noncombat_error = False
    for _ in range(120):
        mask = info.get("action_mask")
        phase = info.get("phase")
        action, shadow = eval_mod.choose_action(
            "hierarchical",
            env,
            obs,
            info,
            mask,
            rng,
            model=None,
            combat_model=model,
            jev_enabled=True,
            jev_adapter=adapter if phase != RunManager.PHASE_COMBAT else _BoomIfCalled(),
        )
        assert mask[action] == 1
        if phase == RunManager.PHASE_COMBAT:
            saw_combat = True
            assert shadow["shadow_status"] == eval_mod.JEV_SHADOW_SKIPPED
            assert _COMBAT_START <= action < _COMBAT_START + _COMBAT_SIZE
        else:
            assert shadow["shadow_status"] == "error"
            saw_noncombat_error = True
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    if not saw_combat:
        pytest.skip("No combat phase reached")
    assert saw_noncombat_error
    assert model.seen_widths
    assert set(model.seen_widths) == {OBS_SIZE}
