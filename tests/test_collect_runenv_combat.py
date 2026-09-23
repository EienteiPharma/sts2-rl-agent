"""Offline hang combat buffer: save/load shapes, collector, train-from-buffer."""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.combat_buffer import (
    BUFFER_VERSION,
    REQUIRED_KEYS,
    CombatReplayEnv,
    collect_transitions,
    collect_worker,
    concat_buffers,
    hang_protocol_meta,
    legal_random_action,
    load_combat_buffer,
    refuse_frozen_path,
    save_combat_buffer,
    split_worker_steps,
    synthetic_combat_buffer,
    validate_buffer,
)
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.runenv_antiforget import (
    SOURCE_LOADOUT,
    SOURCE_RUNENV,
    MixedHangLoadoutEnv,
    MixedHangLoadoutEnvMaker,
    make_loadout_v1_provider,
)
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    RunEnvOnPolicyCombatEnv,
)

_COLLECT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_runenv_combat.py"
_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_combat_from_buffer.py"
_ANTIFORGET_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_combat_runenv_antiforget.py"
)


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collect_mod = _load_mod(_COLLECT_PATH, "collect_runenv_combat")
train_mod = _load_mod(_TRAIN_PATH, "train_combat_from_buffer")


def test_synthetic_save_load_shapes(tmp_path):
    arrays = synthetic_combat_buffer(n=24, n_episodes=6, seed=3)
    validate_buffer(arrays)
    out = tmp_path / "transitions.npz"
    save_combat_buffer(out, arrays, {"policy": "random"})
    loaded, meta = load_combat_buffer(out)
    assert loaded["obs"].shape == (24, OBS_SIZE)
    assert loaded["next_obs"].shape == (24, OBS_SIZE)
    assert loaded["action"].shape == (24,)
    assert loaded["reward"].shape == (24,)
    assert loaded["done"].shape == (24,)
    assert loaded["action_mask"].shape == (24, ACTION_SPACE_SIZE)
    assert loaded["obs"].dtype == np.float32
    assert loaded["action_mask"].dtype == np.int8
    assert meta["buffer_version"] == BUFFER_VERSION
    assert meta["n_transitions"] == 24
    assert meta["obs_size"] == OBS_SIZE
    assert meta["action_size"] == ACTION_SPACE_SIZE
    assert meta["jev_event"] == "off"
    assert meta["jev_neow"] == "off"
    assert meta["start_with_neow"] is True
    assert meta["allows_neow"] is False
    assert meta["policy"] == "random"
    assert set(REQUIRED_KEYS) <= set(loaded)


def test_concat_shards_and_replay_env(tmp_path):
    a = synthetic_combat_buffer(n=8, n_episodes=2, seed=1)
    b = synthetic_combat_buffer(n=8, n_episodes=2, seed=2)
    merged = concat_buffers([a, b])
    assert merged["obs"].shape == (16, OBS_SIZE)
    out = tmp_path / "merged.npz"
    save_combat_buffer(out, merged)
    env = CombatReplayEnv.from_path(out, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert info["replay"] is True
    assert info["jev_event"] == "off"
    assert info["jev_neow"] == "off"
    assert info["start_with_neow"] is True
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert int(mask.sum()) >= 1
    next_obs, reward, terminated, truncated, info2 = env.step(
        int(np.flatnonzero(mask == 1)[0])
    )
    env.close()
    assert next_obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert info2["replay"] is True
    proto = CombatReplayEnv(merged, hang_protocol_meta()).hang_protocol()
    assert proto["allows_event"] is False
    assert proto["allows_neow"] is False


def test_refuse_frozen_buffer_path(tmp_path):
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        refuse_frozen_path("output/combat_ppo_obs_v1_bh_v1/transitions.npz")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        refuse_frozen_path("output/combat_runenv_onpolicy_v1/x.npz")
    ok = refuse_frozen_path(tmp_path / "runenv_combat_buffer" / "transitions.npz")
    assert ok.name == "transitions.npz"


def test_split_worker_steps():
    assert split_worker_steps(10, 1) == [10]
    assert split_worker_steps(10, 4) == [3, 3, 2, 2]
    assert split_worker_steps(5, 4) == [2, 1, 1, 1]
    assert split_worker_steps(0, 3) == [0, 0, 0]


def test_collect_cli_hang_defaults():
    args = collect_mod.parse_args([])
    assert args.policy == "random"
    assert args.n_envs == 1
    assert args.n_steps == 256
    assert args.dry_run is False
    assert "bh_v1" in args.model
    text = _COLLECT_PATH.read_text()
    assert "--jev off" not in text
    assert "HANG_JEV_EVENT" in text
    assert "start_with_neow" in text
    with pytest.raises(SystemExit):
        collect_mod.parse_args(["--policy", "off"])


def test_collect_dry_run_hang_flags():
    report = collect_mod.dry_run(
        collect_mod.parse_args(
            [
                "--dry-run",
                "--out",
                "output/runenv_combat_buffer/dry.npz",
                "--n-envs",
                "4",
                "--n-steps",
                "100",
            ]
        )
    )
    assert report["dry_run"] is True
    assert report["hang"]["jev_event"] == "off"
    assert report["hang"]["jev_neow"] == "off"
    assert report["hang"]["start_with_neow"] is True
    assert report["hang"]["allows_neow"] is False
    assert report["obs_size"] == OBS_SIZE
    assert report["action_size"] == ACTION_SPACE_SIZE
    assert report["worker_quotas"] == [25, 25, 25, 25]
    assert report["start_with_neow"] is True
    assert report["keys"] == list(REQUIRED_KEYS)
    assert report["recommended_n_envs_max"] == 4
    assert "typesafe_key_count" in report
    assert isinstance(report["typesafe_key_count"], int)
    assert report["n_envs_note"] is None
    report16 = collect_mod.dry_run(
        collect_mod.parse_args(
            [
                "--dry-run",
                "--out",
                "output/runenv_combat_buffer/dry16.npz",
                "--n-envs",
                "16",
                "--n-steps",
                "16",
            ]
        )
    )
    assert report16["n_envs_note"] is not None
    assert "2-4" in report16["n_envs_note"]
    dumped = json.dumps(report16)
    assert "TYPESAFE_API_KEY=" not in dumped
    assert "pool-k" not in dumped


def test_collect_dry_run_refuses_bh_v1():
    args = collect_mod.parse_args(
        ["--dry-run", "--out", "output/combat_ppo_obs_v1_bh_v1/transitions.npz"]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        collect_mod.dry_run(args)


def test_collector_write_and_load_shapes(tmp_path):
    env = RunEnvOnPolicyCombatEnv(max_steps=400, max_auto_steps=400)
    rng = np.random.RandomState(7)

    def select_fn(obs, mask):
        assert np.asarray(obs).shape[-1] == OBS_SIZE
        return legal_random_action(mask, rng)

    arrays = collect_transitions(env, 4, rng=rng, select_fn=select_fn, reset_seed=11)
    env.close()
    assert arrays["obs"].shape == (4, OBS_SIZE)
    assert arrays["action_mask"].shape == (4, ACTION_SPACE_SIZE)
    assert arrays["done"].shape == (4,)
    out = tmp_path / "live.npz"
    save_combat_buffer(out, arrays, {"policy": "random", "source": "test"})
    loaded, meta = load_combat_buffer(out)
    assert loaded["obs"].shape == (4, OBS_SIZE)
    assert meta["jev_event"] == HANG_JEV_EVENT == "off"
    assert meta["jev_neow"] == HANG_JEV_NEOW == "off"
    assert meta["start_with_neow"] is HANG_START_WITH_NEOW
    replay = CombatReplayEnv.from_path(out)
    obs, info = replay.reset()
    replay.close()
    assert obs.shape == (OBS_SIZE,)
    assert info["replay"] is True


def test_collect_worker_writes_shard(tmp_path):
    shard = tmp_path / "shard_00.npz"
    row = collect_worker(
        {
            "n_steps": 2,
            "worker_id": 0,
            "seed": 5,
            "policy": "random",
            "model": None,
            "shard": str(shard),
            "max_steps": 400,
        }
    )
    assert row["n_transitions"] == 2
    arrays, meta = load_combat_buffer(shard)
    assert arrays["obs"].shape == (2, OBS_SIZE)
    assert arrays["action_mask"].shape[1] == ACTION_SPACE_SIZE
    assert meta["policy"] == "random"
    pickle.dumps(collect_worker)


def test_mixed_maker_pickle_loadout_half():
    maker = MixedHangLoadoutEnvMaker(seed=3, runenv_frac=0.0, max_steps=40)
    loaded = pickle.loads(pickle.dumps(maker))
    env = loaded()
    obs, info = env.reset(seed=1)
    env.close()
    assert info["mix_source"] == SOURCE_LOADOUT
    assert obs.shape == (OBS_SIZE,)


def test_mixed_maker_pickle_buffer_path(tmp_path):
    arrays = synthetic_combat_buffer(n=12, n_episodes=3, seed=0)
    path = tmp_path / "buf.npz"
    save_combat_buffer(path, arrays)
    maker = MixedHangLoadoutEnvMaker(
        seed=1, runenv_frac=1.0, max_steps=40, buffer_path=str(path)
    )
    loaded = pickle.loads(pickle.dumps(maker))
    env = loaded()
    obs, info = env.reset(seed=2)
    env.close()
    assert obs.shape == (OBS_SIZE,)
    assert info["mix_source"] == SOURCE_RUNENV
    assert info.get("replay") is True


def test_mix_replay_half_no_live_runenv():
    arrays = synthetic_combat_buffer(n=10, n_episodes=2, seed=4)
    replay = CombatReplayEnv(arrays, hang_protocol_meta(), seed=0)
    env = MixedHangLoadoutEnv(
        runenv_frac=1.0,
        loadout_provider=make_loadout_v1_provider(offset=0),
        max_steps=40,
        runenv_env=replay,
    )
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    obs2, reward, _t, _tr, info2 = env.step(int(np.flatnonzero(mask == 1)[0]))
    env.close()
    assert info["mix_source"] == SOURCE_RUNENV
    assert info.get("replay") is True
    assert obs.shape == (OBS_SIZE,)
    assert obs2.shape == (OBS_SIZE,)
    assert info2.get("replay") is True
    assert isinstance(reward, float)


def test_train_from_buffer_cli_and_dry_run(tmp_path):
    args = train_mod.parse_args([])
    assert args.continue_from == train_mod.HUNG_COMBAT_ZIP
    assert "bh_v1" in args.continue_from
    assert args.runenv_frac == pytest.approx(0.7)
    assert args.n_envs == 1
    assert args.total_timesteps == 2048
    assert args.output_dir == train_mod.DEFAULT_OUTPUT_DIR
    assert args.device == "auto"
    report = train_mod.dry_run(
        train_mod.parse_args(
            ["--dry-run", "--output-dir", "output/combat_runenv_offline_dry"]
        )
    )
    assert report["dry_run"] is True
    assert report["buffer_source"] == "synthetic"
    assert report["jev_on_learn_path"] is False
    assert report["device_requested"] == "auto"
    assert report["device"] in ("cpu", "cuda")
    assert report["hang"]["jev_event"] == "off"
    assert report["hang"]["jev_neow"] == "off"
    assert report["hang"]["start_with_neow"] is True
    assert report["buffer_reset"]["obs_size"] == OBS_SIZE
    assert report["buffer_reset"]["replay"] is True
    assert report["mix_buffer_reset"]["replay"] is True
    assert report["loadout_reset"]["mix_source"] == "loadout_v1"
    assert report["loadout_reset"]["obs_size"] == OBS_SIZE
    assert report["online_antiforget_still_valid"] is True
    assert "combat_ppo_obs_v1_bh_v1" in report["frozen_outdirs"]
    arrays = synthetic_combat_buffer(n=8, n_episodes=2, seed=1)
    buf = tmp_path / "transitions.npz"
    save_combat_buffer(buf, arrays)
    report2 = train_mod.dry_run(
        train_mod.parse_args(
            [
                "--dry-run",
                "--buffer",
                str(buf),
                "--output-dir",
                str(tmp_path / "offline_out"),
            ]
        )
    )
    assert report2["buffer_source"] == "disk"
    assert report2["n_transitions"] == 8


def test_train_from_buffer_resolve_device():
    assert train_mod.parse_args(["--device", "cpu"]).device == "cpu"
    assert train_mod.parse_args(["--device", "cuda"]).device == "cuda"
    assert train_mod.parse_args(["--device", "cuda:0"]).device == "cuda:0"
    assert train_mod.resolve_device("auto", cuda_available=False) == "cpu"
    assert train_mod.resolve_device("auto", cuda_available=True) == "cuda"
    assert train_mod.resolve_device("cpu", cuda_available=True) == "cpu"
    assert train_mod.resolve_device("cuda", cuda_available=True) == "cuda"
    assert train_mod.resolve_device("cuda:1", cuda_available=True) == "cuda:1"
    with pytest.raises(SystemExit, match="cuda requested"):
        train_mod.resolve_device("cuda", cuda_available=False)
    with pytest.raises(SystemExit, match="unknown"):
        train_mod.resolve_device("tpu", cuda_available=False)


def test_train_from_buffer_refuses_frozen():
    args = train_mod.parse_args(
        ["--dry-run", "--output-dir", "output/combat_ppo_obs_v1_bh_v1"]
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_mod.dry_run(args)


def test_train_from_buffer_requires_buffer_without_dry_run():
    args = train_mod.parse_args(["--output-dir", "output/combat_runenv_offline_x"])
    with pytest.raises(SystemExit, match="--buffer"):
        train_mod.train(args)


def test_antiforget_n_envs_subproc_documented():
    spec = importlib.util.spec_from_file_location(
        "train_combat_runenv_antiforget_offline", _ANTIFORGET_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--n-envs", "4", "--dry-run", "--output-dir", "output/af_n"])
    assert args.n_envs == 4
    report = mod.dry_run(args)
    assert report["n_envs_backend"] == "SubprocVecEnv"
    assert report["subproc_maker"] == "MixedHangLoadoutEnvMaker"
    maker = mod.make_masked_env(0, runenv_frac=0.0, max_steps=40)
    pickle.dumps(maker)
