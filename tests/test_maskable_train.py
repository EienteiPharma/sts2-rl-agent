"""Tests for the shared owner of masked train envs, the eval callback, and the model."""

import inspect
import pickle
import re
import subprocess
import sys
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch

from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.train.curriculum import TrainHyperparams
from sts2_env.train.maskable import (
    RunEnvSpec,
    _CachedMaskFactory,
    _InfoMaskEnv,
    _ParentMaskVec,
    MissingActionMask,
    open_combat_train_surface,
    open_run_train_surface,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = (
    REPO_ROOT / "scripts" / "train_full_run.py",
    REPO_ROOT / "scripts" / "train_combat.py",
)


def _accelerator_launches_kernels() -> bool:
    """A visible device is not a working one.

    This box exposes an iGPU at ``cuda:0`` that aborts the whole process on its
    first kernel, which no in-process guard survives. Probe in a child.
    """
    probe = "import torch; t=torch.ones(8, device='cuda'); print(float((t*2).sum()))"
    try:
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


requires_working_accelerator = pytest.mark.skipif(
    not (torch.cuda.is_available() and _accelerator_launches_kernels()),
    reason="no accelerator that can launch a kernel is visible",
)


def _hyperparams() -> TrainHyperparams:
    return TrainHyperparams(
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.0,
    )


class _MaskScriptedEnv(gymnasium.Env):
    """Bare env whose mask names the step it came from, so staleness is visible."""

    def __init__(self, masks: list[np.ndarray], done_at: int) -> None:
        self.action_space = gymnasium.spaces.Discrete(4)
        self.observation_space = gymnasium.spaces.Box(
            -1.0, 1.0, (2,), dtype=np.float32
        )
        self._masks = masks
        self._done_at = done_at
        self._i = 0

    def _mask(self) -> np.ndarray:
        return self._masks[self._i].copy()

    def reset(self, **kwargs):
        self._i = 0
        return np.zeros(2, dtype=np.float32), {"action_mask": self._mask()}

    def step(self, action):
        self._i = min(self._i + 1, len(self._masks) - 1)
        done = self._i >= self._done_at
        return (
            np.zeros(2, dtype=np.float32),
            0.0,
            done,
            False,
            {"action_mask": self._mask()},
        )

    def action_masks(self) -> np.ndarray:
        raise AssertionError("the cache owns the mask, the env must not be asked")


def _one_hot(index: int) -> np.ndarray:
    mask = np.zeros(4, dtype=np.int8)
    mask[index] = 1
    return mask


def test_run_vec_env_stacks_int8_masks():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([RunEnvSpec(act_count=1, reward_shaping=True)] * 2)
    try:
        masks = get_action_masks(env)
    finally:
        env.close()

    assert masks.shape == (2, 157)
    assert masks.dtype == np.int8


def test_combat_vec_env_stacks_int8_masks():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: STS2CombatEnv()] * 2)
    try:
        masks = get_action_masks(env)
    finally:
        env.close()

    assert masks.shape == (2, 115)
    assert masks.dtype == np.int8


@pytest.mark.parametrize("script", TRAIN_SCRIPTS, ids=lambda path: path.name)
def test_train_scripts_leave_masking_to_the_owner(script):
    text = script.read_text(encoding="utf-8")

    assert re.search(r"\bActionMasker\b", text) is None
    assert re.search(r"(?<!Maskable)EvalCallback", text) is None
    assert re.search(r"\bMaskablePPO\s*\(", text) is None
    assert "build_train_model" not in text


def test_mask_cache_serves_the_post_reset_mask_after_a_done_step():
    """VecEnv auto-resets a finished env; the cache must follow the new episode."""
    pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    opening = _one_hot(0)
    terminal = _one_hot(3)
    env = DummyVecEnv([_CachedMaskFactory(lambda: _MaskScriptedEnv([opening, terminal], done_at=1))])
    try:
        env.reset()
        assert get_action_masks(env)[0].tolist() == opening.tolist()

        env.step(np.array([0]))
        masks = get_action_masks(env)

        assert masks[0].tolist() == opening.tolist()
        assert masks[0].tolist() != terminal.tolist()
    finally:
        env.close()


def test_parent_mask_vec_serves_the_post_reset_mask_without_rpc():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    opening = _one_hot(0)
    terminal = _one_hot(3)
    env = _ParentMaskVec(
        DummyVecEnv(
            [_CachedMaskFactory(lambda: _MaskScriptedEnv([opening, terminal], done_at=1))]
        )
    )
    try:
        env.reset()

        def boom() -> np.ndarray:
            raise AssertionError("parent cache must not RPC action_masks")

        env.venv.envs[0].action_masks = boom
        assert get_action_masks(env)[0].tolist() == opening.tolist()

        env.step(np.array([0]))
        masks = get_action_masks(env)
        assert masks[0].tolist() == opening.tolist()
        assert masks[0].tolist() != terminal.tolist()
    finally:
        env.close()


def test_missing_mask_on_reset_raises():
    class _NoMask(gymnasium.Env):
        def reset(self, **kwargs):
            return np.zeros(2, dtype=np.float32), {}

    with pytest.raises(MissingActionMask):
        _InfoMaskEnv(_NoMask()).reset()


def test_missing_mask_on_step_raises():
    class _NoMask(gymnasium.Env):
        action_space = STS2CombatEnv().action_space

        def reset(self, **kwargs):
            return np.zeros(2, dtype=np.float32), {
                "action_mask": np.ones(self.action_space.n, dtype=np.int8)
            }

        def step(self, action):
            return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    env = _InfoMaskEnv(_NoMask())
    env.reset()
    with pytest.raises(MissingActionMask):
        env.step(0)


def test_short_mask_raises():
    class _ShortMask(gymnasium.Env):
        action_space = STS2CombatEnv().action_space

        def reset(self, **kwargs):
            return np.zeros(2, dtype=np.float32), {"action_mask": np.ones(3, dtype=np.int8)}

    with pytest.raises(MissingActionMask):
        _InfoMaskEnv(_ShortMask()).reset()


def test_action_masks_returns_a_copy():
    env = _InfoMaskEnv(_MaskScriptedEnv([_one_hot(0)], done_at=99))
    env.reset()
    first = env.action_masks()
    first[:] = 7

    assert env.action_masks().tolist() == _one_hot(0).tolist()


def test_action_masks_before_reset_raises():
    env = _InfoMaskEnv(_MaskScriptedEnv([_one_hot(0)], done_at=99))

    with pytest.raises(MissingActionMask, match="before reset"):
        env.action_masks()


def test_cached_mask_factory_survives_pickling():
    factory = _CachedMaskFactory(RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20))
    restored = pickle.loads(pickle.dumps(factory))

    env = restored()
    try:
        env.reset()
        assert env.action_masks().shape == (157,)
    finally:
        env.close()


def test_open_run_surface_returns_a_masked_callback(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.maskable.utils import get_action_masks

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=50),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        masks = get_action_masks(job.train_env)
        assert masks.shape == (1, 157)
        assert masks.dtype == np.int8
        assert type(job.eval_callback).__name__ == "MaskableEvalCallback"
        assert job.eval_callback.use_masking is True
        assert job.eval_callback.deterministic is False
    finally:
        job.close()

    job.close()


def test_collect_trunk_binds_forward_and_unbinds_on_close(tmp_path):
    pytest.importorskip("sb3_contrib")

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        policy = job.model.policy
        trunk = policy._collect_trunk
        assert trunk.batch == 1
        assert not inspect.ismethod(policy.forward)
        assert trunk.state.reason in ("cpu", "uncaptured")
    finally:
        job.close()
    assert inspect.ismethod(job.model.policy.forward)


@requires_working_accelerator
def test_collect_trunk_captures_or_falls_back_after_learn(tmp_path):
    pytest.importorskip("sb3_contrib")

    from sts2_env.train.maskable import _TrunkEager, _TrunkLive

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        job.learn(total_timesteps=8, progress_bar=False)
        trunk = job.model.policy._collect_trunk
        assert inspect.ismethod(job.model.policy.evaluate_actions)
        if isinstance(trunk.state, _TrunkLive):
            assert trunk.state.batch == 1
            assert trunk.state.logits.device.type == "cuda"
        else:
            assert isinstance(trunk.state, _TrunkEager)
            assert trunk.state.reason == "capture_failed"
    finally:
        job.close()
    assert inspect.ismethod(job.model.policy.forward)


def test_open_surfaces_start_on_the_rollout_device(tmp_path):
    pytest.importorskip("sb3_contrib")
    import torch

    from sts2_env.train import maskable

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        target = maskable._rollout_device()
        assert job.model.device.type == target.type
        assert job.model.policy.device.type == target.type
    finally:
        job.close()


def test_combat_surface_keeps_the_default_net_arch(tmp_path):
    pytest.importorskip("sb3_contrib")

    job = open_combat_train_surface(
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        policy = job.model.policy
        assert policy.mlp_extractor.policy_net[0].out_features == 64
        assert policy.mlp_extractor.value_net[0].out_features == 64
    finally:
        job.close()


def test_run_surface_uses_the_net_arch_constant(tmp_path):
    pytest.importorskip("sb3_contrib")

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        assert job.model.policy.mlp_extractor.policy_net[0].out_features == 256
    finally:
        job.close()


def test_learn_writes_steps_and_returns_the_model_to_cpu(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    _cpu_updates(monkeypatch)

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        job.learn(total_timesteps=8, progress_bar=False)
        assert job.model.num_timesteps == 8
        assert job.model.device == torch.device("cpu")
    finally:
        job.close()


def test_pin_is_idempotent_on_cpu(tmp_path, monkeypatch):
    """A second pin must not rebuild Adam state or move anything."""
    pytest.importorskip("sb3_contrib")
    _cpu_updates(monkeypatch)

    from sts2_env.train.maskable import _pin_algorithm_device

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        job.learn(total_timesteps=8, progress_bar=False)
        assert job.model.policy.optimizer.state, "one step must leave Adam state"

        def snapshot():
            state = job.model.policy.optimizer.state
            return [
                (tensor.dtype, tensor.device, id(tensor))
                for entry in state.values()
                for tensor in entry.values()
                if isinstance(tensor, torch.Tensor)
            ]

        before = snapshot()
        _pin_algorithm_device(job.model, "cpu")

        assert snapshot() == before
        assert job.model.device == torch.device("cpu")
        assert job.model.rollout_buffer.device == torch.device("cpu")
    finally:
        job.close()


@requires_working_accelerator
def test_learn_keeps_the_model_on_the_accelerator(tmp_path):
    pytest.importorskip("sb3_contrib")

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        job.learn(total_timesteps=8, progress_bar=False)
        assert job.model.device.type == "cuda"
        assert all(p.device.type == "cuda" for p in job.model.policy.parameters())
    finally:
        job.close()


@requires_working_accelerator
def test_pin_round_trips_through_the_accelerator(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")

    from sts2_env.train.maskable import _pin_algorithm_device

    job = open_run_train_surface(
        RunEnvSpec(act_count=1, reward_shaping=True, max_steps=20),
        n_envs=1,
        output_dir=tmp_path,
        eval_freq=10,
        n_eval_episodes=1,
        start=_fresh_start(),
        hyperparams=_hyperparams(),
    )
    try:
        _cpu_updates(monkeypatch)
        job.learn(total_timesteps=8, progress_bar=False)

        _pin_algorithm_device(job.model, "cuda")
        assert job.model.device.type == "cuda"
        assert all(p.device.type == "cuda" for p in job.model.policy.parameters())
        moments = [
            tensor
            for entry in job.model.policy.optimizer.state.values()
            for tensor in entry.values()
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 0
        ]
        steps = [
            tensor
            for entry in job.model.policy.optimizer.state.values()
            for tensor in entry.values()
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 0
        ]
        assert moments
        assert all(tensor.device.type == "cuda" for tensor in moments)
        assert steps
        assert all(tensor.device.type == "cpu" for tensor in steps)
        assert job.model.rollout_buffer.device.type == "cuda"

        _pin_algorithm_device(job.model, "cpu")
        assert job.model.device.type == "cpu"
        assert all(p.device.type == "cpu" for p in job.model.policy.parameters())
        assert job.model.rollout_buffer.device.type == "cpu"
    finally:
        job.close()


def _fresh_start():
    from sts2_env.train.curriculum import FreshStart

    return FreshStart()


def _cpu_updates(monkeypatch):
    """Keep the update on CPU so a unit test never needs a working accelerator."""
    from sts2_env.train import maskable

    monkeypatch.setattr(maskable, "_update_device", lambda: torch.device("cpu"))
