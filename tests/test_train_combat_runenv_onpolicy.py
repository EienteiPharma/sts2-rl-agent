"""RunEnv on-policy combat micro: hang flags, combat-only env, no fixtures."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, _COMBAT_START
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    RunEnvOnPolicyCombatEnv,
    hang_jev_flags,
    to_runenv_combat_action,
)
from sts2_env.run.run_manager import RunManager

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_combat_runenv_onpolicy.py"


def _load_train_mod():
    spec = importlib.util.spec_from_file_location(
        "train_combat_runenv_onpolicy", _TRAIN_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


train_mod = _load_train_mod()


class FakeSpace:
    def __init__(self, dim: int):
        self.shape = (dim,)


def test_script_has_no_loadout_cli():
    args = train_mod.parse_args([])
    assert not hasattr(args, "loadout")
    with pytest.raises(SystemExit):
        train_mod.parse_args(["--loadout", "mix_neow_v1"])
    text = _TRAIN_PATH.read_text()
    assert "--loadout" not in text
    assert "mix_neow_v1" in text  # mentioned as forbidden
    assert "LOADOUT_SUITES" not in text


def test_cli_hang_defaults_and_continue_from():
    args = train_mod.parse_args([])
    assert args.continue_from == train_mod.HUNG_COMBAT_ZIP
    assert "combat_ppo_obs_v1_bh_v1" in args.continue_from
    assert args.output_dir == train_mod.DEFAULT_OUTPUT_DIR
    assert args.n_envs == 1
    assert args.total_timesteps == 2048
    assert args.n_steps == 128
    assert args.dry_run is False
    alias = train_mod.parse_args(["--model", "/tmp/combat.zip"])
    assert alias.continue_from == "/tmp/combat.zip"
    cont = train_mod.parse_args(["--continue-from", "/tmp/other.zip"])
    assert cont.continue_from == "/tmp/other.zip"


def test_hang_protocol_flags_frozen():
    flags = hang_jev_flags()
    assert HANG_JEV == "on"
    assert HANG_JEV_EVENT == "off"
    assert HANG_JEV_NEOW == "off"
    assert HANG_START_WITH_NEOW is True
    assert flags.allows_event() is False
    assert flags.allows_neow() is False
    proto = RunEnvOnPolicyCombatEnv(max_steps=20).hang_protocol()
    assert proto["start_with_neow"] is True
    assert proto["jev_event"] == "off"
    assert proto["jev_neow"] == "off"
    assert proto["allows_neow"] is False
    assert proto["combat_obs_size"] == OBS_SIZE == 181


def test_refuse_overwrite_bh_v1(tmp_path):
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.refuse_overwrite_hung_zip("output/combat_ppo_obs_v1_bh_v1")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.refuse_overwrite_hung_zip(
            "/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1"
        )
    ok = train_mod.refuse_overwrite_hung_zip(tmp_path / "combat_runenv_onpolicy_v1")
    assert ok.name == "combat_runenv_onpolicy_v1"


def test_continue_from_path_and_obs_dim(tmp_path):
    missing = tmp_path / "nope.zip"
    with pytest.raises(SystemExit, match="not found"):
        train_mod.resolve_continue_from(str(missing), must_exist=True)
    fake = tmp_path / "final_model.zip"
    fake.write_bytes(b"not-a-real-zip")
    assert train_mod.resolve_continue_from(str(fake), must_exist=True) == fake
    good = SimpleNamespace(observation_space=FakeSpace(OBS_SIZE))
    assert train_mod.require_combat_obs_dim(good) == OBS_SIZE
    runenv = SimpleNamespace(observation_space=FakeSpace(RUN_OBS_SIZE))
    with pytest.raises(SystemExit, match="OBS_SIZE"):
        train_mod.require_combat_obs_dim(runenv)


def test_train_deps_guard_without_learn():
    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        with pytest.raises(SystemExit, match="sb3"):
            train_mod.require_train_deps()
        return
    pytest.skip("sb3-contrib present; learn() still not invoked")


def test_combat_only_env_constructs_and_reaches_combat():
    env = RunEnvOnPolicyCombatEnv(max_steps=400, max_auto_steps=400)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_SIZE,)
    assert obs.shape[-1] != RUN_OBS_SIZE
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert int(mask.sum()) >= 1
    assert info["start_with_neow"] is True
    assert info["jev_event"] == "off"
    assert info["jev_neow"] == "off"
    assert info["loadout"] is None
    assert info["combat_obs_size"] == OBS_SIZE
    if info.get("combat_unreachable"):
        env.close()
        pytest.skip("reset did not reach combat")
    assert info["phase"] == RunManager.PHASE_COMBAT
    assert int(info["noncombat_auto_steps"]) >= 1
    assert int(info["combat_steps"]) == 0
    local = int(np.flatnonzero(mask == 1)[0])
    assert to_runenv_combat_action(local) == _COMBAT_START + local
    obs2, reward, terminated, truncated, info2 = env.step(local)
    env.close()
    assert obs2.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert info2["combat_steps"] == 1
    assert info2["loadout"] is None
    assert terminated in (True, False)
    assert truncated in (True, False)


def test_dry_run_refuses_bh_v1_outdir():
    args = train_mod.parse_args(
        ["--dry-run", "--output-dir", "output/combat_ppo_obs_v1_bh_v1"]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.dry_run(args)


def test_dry_run_wires_hang_and_skips_learn():
    report = train_mod.dry_run(
        train_mod.parse_args(
            [
                "--dry-run",
                "--output-dir",
                "output/combat_runenv_onpolicy_dry",
                "--total-timesteps",
                "256",
            ]
        )
    )
    assert report["dry_run"] is True
    assert report["loadout_forbidden"] is True
    assert report["loadout"] is None
    assert report["hang"]["start_with_neow"] is True
    assert report["hang"]["jev_event"] == "off"
    assert report["hang"]["jev_neow"] == "off"
    assert report["hang"]["allows_event"] is False
    assert report["hang"]["allows_neow"] is False
    assert report["obs_size"] == OBS_SIZE
    assert report["action_size"] == ACTION_SPACE_SIZE
    assert report["obs_shape"] == [OBS_SIZE]
    assert "combat_ppo_obs_v1_bh_v1" not in report["output_dir"]


def test_make_onpolicy_env_has_no_loadout_provider():
    factory = train_mod.make_onpolicy_env(seed=1, max_steps=40)
    env = factory()
    assert isinstance(env, RunEnvOnPolicyCombatEnv)
    assert not hasattr(env, "loadout_provider")
    env.close()
