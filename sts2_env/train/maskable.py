"""One owner for the vec envs, the masked eval callback, and the MaskablePPO a trainer uses.

Stock ``EvalCallback`` calls ``predict`` with no masks, so a run that evaluated
through it scored a different agent from the one it trained. This module hands
back the train vec env, the eval vec env, a masking eval callback, and the
algorithm as one value, so no caller can build a bare env pair, attach an
unmasked callback, or pick its own device.

Predict and the PPO update share the accelerator when one is visible.
CPU predict of ``CardEmbedExtractor`` saturates the parent at batch 16.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import gymnasium
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

from sts2_env.train.curriculum import (
    CLIP_RANGE,
    GAE_LAMBDA,
    NET_ARCH,
    FreshStart,
    TrainHyperparams,
    TrainStart,
    load_maskable_checkpoint,
)

RUN_TRAIN_MAX_STEPS = 2000
TRAINER_TORCH_THREADS = 4

def _rollout_device() -> torch.device:
    """Same device as the PPO update.

    A cheap MLP used to be faster on CPU. The 2443-wide extractor is not.
    Tests patch ``_update_device`` and inherit this.
    """
    return _update_device()

__all__ = [
    "RunEnvSpec",
    "MaskedTrainSurface",
    "open_run_train_surface",
    "open_combat_train_surface",
]


@dataclass(frozen=True)
class RunEnvSpec:
    """Picklable factory for a bare ``STS2RunEnv``.

    Workers unpickle the spec and call it, so the spawn payload carries
    primitives instead of a closure over the caller's locals.
    """

    act_count: int
    reward_shaping: bool
    max_steps: int = RUN_TRAIN_MAX_STEPS
    reward_win: float | None = None
    reward_hp_scale: float | None = None
    reward_hallway: float | None = None
    reward_elite: float | None = None
    reward_act: float | None = None
    reward_combat: float | None = None
    reward_boss_scale: float | None = None
    reward_boss_hp_scale: float | None = None
    reward_boss_block_scale: float | None = None
    reward_relic_scale: float | None = None
    reward_map_relic_scale: float | None = None
    reward_boss_potion_scale: float | None = None
    reward_boss_waste_scale: float | None = None
    reward_anger_pick_scale: float | None = None
    reward_early_elite_scale: float | None = None
    obs_layout: str = "v2"
    boss_start_prob: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.act_count <= 3:
            raise ValueError(f"act_count must be 1..3, got {self.act_count}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if not 0.0 <= float(self.boss_start_prob) <= 1.0:
            raise ValueError(
                f"boss_start_prob must be in [0, 1], got {self.boss_start_prob!r}"
            )

    def __call__(self) -> Any:
        import multiprocessing as mp
        import os

        from sts2_env.gym_env.run_env import STS2RunEnv

        if mp.current_process().name != "MainProcess":
            os.environ["HIP_VISIBLE_DEVICES"] = ""
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return STS2RunEnv(
            act_count=self.act_count,
            reward_shaping=self.reward_shaping,
            max_steps=self.max_steps,
            reward_win=self.reward_win,
            reward_hp_scale=self.reward_hp_scale,
            reward_hallway=self.reward_hallway,
            reward_elite=self.reward_elite,
            reward_act=self.reward_act,
            reward_combat=self.reward_combat,
            reward_boss_scale=self.reward_boss_scale,
            reward_boss_hp_scale=self.reward_boss_hp_scale,
            reward_boss_block_scale=self.reward_boss_block_scale,
            reward_relic_scale=self.reward_relic_scale,
            reward_map_relic_scale=self.reward_map_relic_scale,
            reward_boss_potion_scale=self.reward_boss_potion_scale,
            reward_boss_waste_scale=self.reward_boss_waste_scale,
            reward_anger_pick_scale=self.reward_anger_pick_scale,
            reward_early_elite_scale=self.reward_early_elite_scale,
            obs_layout=self.obs_layout,
            boss_start_prob=self.boss_start_prob,
        )


class MaskedTrainSurface:
    """One job: cached train/eval vecs, masked eval callback, hybrid MaskablePPO."""

    train_env: Any
    eval_env: Any
    eval_callback: Any
    model: Any

    def __init__(
        self,
        train_env: Any,
        eval_env: Any,
        eval_callback: Any,
        model: Any,
    ) -> None:
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_callback = eval_callback
        self.model = model
        self._collect_trunk = getattr(model.policy, "_collect_trunk", None)
        self._closed = False

    def learn(
        self,
        total_timesteps: int,
        *,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train through the surface so the device split cannot be omitted."""
        callbacks = CallbackList([_HybridDeviceCallback(), self.eval_callback])
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=progress_bar,
                reset_num_timesteps=reset_num_timesteps,
            )
        finally:
            _pin_algorithm_device(self.model, _rollout_device())

    def close(self) -> None:
        if self._closed:
            return
        if self._collect_trunk is not None:
            self._collect_trunk.unbind()
        self.train_env.close()
        self.eval_env.close()
        self._closed = True


def open_run_train_surface(
    spec: RunEnvSpec,
    n_envs: int,
    output_dir: Path,
    eval_freq: int,
    n_eval_episodes: int,
    start: TrainStart,
    hyperparams: TrainHyperparams,
) -> MaskedTrainSurface:
    """Full-run job. ``start`` and ``hyperparams`` are required so a model always exists."""
    return _open_surface(
        spec,
        _Parallelism.parse(n_envs),
        _EvalSchedule.parse(eval_freq, n_eval_episodes),
        output_dir,
        start,
        hyperparams,
        net_arch=NET_ARCH,
    )


def open_combat_train_surface(
    n_envs: int,
    output_dir: Path,
    eval_freq: int,
    n_eval_episodes: int,
    start: TrainStart,
    hyperparams: TrainHyperparams,
) -> MaskedTrainSurface:
    """Combat job. Same contract as the run surface, on combat envs."""
    return _open_surface(
        _CombatEnvFactory(),
        _Parallelism.parse(n_envs),
        _EvalSchedule.parse(eval_freq, n_eval_episodes),
        output_dir,
        start,
        hyperparams,
        net_arch=None,
    )


@dataclass(frozen=True)
class _CombatEnvFactory:
    def __call__(self) -> Any:
        from sts2_env.gym_env.combat_env import STS2CombatEnv

        return STS2CombatEnv()


@dataclass(frozen=True)
class _CachedMaskFactory:
    """Picklable env_fn. Workers unpickle this and call it, so no lambdas."""

    inner: Any

    def __call__(self) -> "_InfoMaskEnv":
        return _InfoMaskEnv(self.inner())


class MissingActionMask(KeyError):
    """An env returned no usable ``action_mask``, so masking cannot be trusted."""


class _InfoMaskEnv(gymnasium.Wrapper):
    """Serve the mask from ``info`` so a step computes it once.

    ``MaskablePPO`` asks for masks before every step, and the STS2 env already
    builds one per step for ``info``. Re-deriving it on demand doubles that
    work. The cache lives here, below VecEnv auto-reset, so the mask after a
    done step is the new episode's, not the terminal one in ``buf_infos``.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._mask: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._mask = _require_action_mask(info, self, where="reset")
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._mask = _require_action_mask(info, self, where="step")
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._mask is None:
            raise MissingActionMask(
                f"{type(self.env.unwrapped).__name__} was asked for a mask before reset"
            )
        return self._mask.copy()


class _ParentMaskVec(VecEnvWrapper):
    """Serve masks from the last step infos so collect does not RPC workers.

    ``MaskablePPO`` calls ``env_method('action_masks')`` before every step.
    Subproc workers already put that array in ``info`` and, after auto-reset,
    in ``reset_infos``. A done step's ``info['action_mask']`` is the terminal
    episode. The new episode's mask is ``reset_infos[i]``.
    """

    def __init__(self, venv: Any) -> None:
        super().__init__(venv)
        self._masks: list[np.ndarray] | None = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self._masks = [
            _require_action_mask(info, self, where="reset_infos")
            for info in self.venv.reset_infos
        ]
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        masks: list[np.ndarray] = []
        reset_infos = self.venv.reset_infos
        for i, done in enumerate(dones):
            info = reset_infos[i] if done else infos[i]
            where = "reset_infos" if done else "step"
            masks.append(_require_action_mask(info, self, where=where))
        self._masks = masks
        return obs, rewards, dones, infos

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if (
            method_name == "action_masks"
            and not method_args
            and not method_kwargs
            and self._masks is not None
        ):
            idxs = list(self.venv._get_indices(indices))
            return [self._masks[i].copy() for i in idxs]
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )


@dataclass(frozen=True)
class _TrunkEager:
    reason: Literal["cpu", "capture_failed", "uncaptured"]


@dataclass
class _TrunkLive:
    graph: Any
    static_obs: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor
    storage_ids: tuple[int, ...]
    batch: int


_TrunkState = _TrunkEager | _TrunkLive

_CAPTURE_ERRORS: tuple[type[BaseException], ...] = (RuntimeError,)
_acc_err = getattr(torch, "AcceleratorError", None)
if isinstance(_acc_err, type):
    _CAPTURE_ERRORS = (RuntimeError, _acc_err)


def _storage_ids(policy: Any) -> tuple[int, ...]:
    return tuple(
        t.data_ptr() for t in (*policy.parameters(), *policy.buffers())
    )


def _tensor_trunk(
    policy: Any, obs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    values = policy.value_net(latent_vf)
    logits = policy.action_net(latent_pi)
    return logits, values


def _eager_masked_dist(
    policy: Any,
    logits: torch.Tensor,
    values: torch.Tensor,
    deterministic: bool,
    action_masks: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution = policy.action_dist.proba_distribution(action_logits=logits)
    if action_masks is not None:
        distribution.apply_masking(action_masks)
    actions = distribution.get_actions(deterministic=deterministic)
    log_prob = distribution.log_prob(actions)
    actions = actions.reshape((-1, *policy.action_space.shape))
    return actions, values, log_prob


class _CollectTrunk:
    """CUDA graph of the tensor trunk, used only from collect ``policy()``.

    Train calls ``evaluate_actions`` at batch 4096. Collect calls
    ``forward`` at ``n_envs``. Wrapping the instance ``forward`` keeps
    those paths apart. Categorical mask and sample stay eager. HIP
    capture of full ``forward`` failed with stream capture invalidated.
    """

    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.state: _TrunkState = _TrunkEager(reason="cpu")
        self._eager_forward: Any = None
        self._policy: Any = None

    def bind(self, policy: Any) -> None:
        if getattr(policy, "_collect_trunk", None) is self:
            return
        self._policy = policy
        self._eager_forward = type(policy).forward
        trunk = self

        def collect_forward(
            obs: Any,
            deterministic: bool = False,
            action_masks: np.ndarray | None = None,
        ) -> tuple[Any, Any, Any]:
            return trunk.collect_forward(
                policy, obs, deterministic, action_masks
            )

        policy.forward = collect_forward
        policy._collect_trunk = self
        if torch.cuda.is_available():
            self.state = _TrunkEager(reason="uncaptured")

    def unbind(self) -> None:
        policy = self._policy
        if policy is None:
            return
        if getattr(policy, "_collect_trunk", None) is self:
            try:
                del policy.forward
            except AttributeError:
                pass
            try:
                del policy._collect_trunk
            except AttributeError:
                pass
        self._policy = None

    def mark_stale(self) -> None:
        if isinstance(self.state, _TrunkLive):
            self.state = _TrunkEager(reason="uncaptured")

    def collect_forward(
        self,
        policy: Any,
        obs: Any,
        deterministic: bool,
        action_masks: np.ndarray | None,
    ) -> tuple[Any, Any, Any]:
        eager = self._eager_forward
        if (
            policy.training
            or not torch.is_tensor(obs)
            or int(obs.shape[0]) != self.batch
            or not torch.cuda.is_available()
            or obs.device.type != "cuda"
        ):
            return eager(
                policy,
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        failed = (
            isinstance(self.state, _TrunkEager)
            and self.state.reason == "capture_failed"
        )
        live = isinstance(self.state, _TrunkLive)
        if live and self.state.storage_ids != _storage_ids(policy):
            live = False
        if not live and not failed:
            self.state = self._capture(policy, obs)
            live = isinstance(self.state, _TrunkLive)
        if live:
            self.state.static_obs.copy_(obs)
            self.state.graph.replay()
            return _eager_masked_dist(
                policy,
                self.state.logits,
                self.state.values,
                deterministic,
                action_masks,
            )
        logits, values = _tensor_trunk(policy, obs)
        return _eager_masked_dist(
            policy, logits, values, deterministic, action_masks
        )

    def _capture(self, policy: Any, obs: torch.Tensor) -> _TrunkState:
        try:
            static_obs = obs.detach().clone()
            with torch.no_grad():
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        logits, values = _tensor_trunk(policy, static_obs)
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits, values = _tensor_trunk(policy, static_obs)
            return _TrunkLive(
                graph=graph,
                static_obs=static_obs,
                logits=logits,
                values=values,
                storage_ids=_storage_ids(policy),
                batch=self.batch,
            )
        except _CAPTURE_ERRORS as exc:
            warnings.warn(
                f"collect CUDA graph capture failed, using eager trunk: {exc}",
                stacklevel=2,
            )
            return _TrunkEager(reason="capture_failed")


def _install_collect_trunk(model: Any, *, batch: int) -> None:
    trunk = _CollectTrunk(batch)
    trunk.bind(model.policy)


def _require_action_mask(
    info: dict[str, Any], env: Any, *, where: str
) -> np.ndarray:
    """Copy the env's mask into the cache, or fail closed.

    A missing or short mask would let ``MaskablePPO`` sample an illegal
    action, which corrupts the rollout silently, so both raise here.
    """
    name = type(env.unwrapped).__name__
    if "action_mask" not in info:
        raise MissingActionMask(f"{name} returned no action_mask from {where}")

    mask = np.asarray(info["action_mask"], dtype=np.int8).copy()
    expected = env.action_space.n
    if mask.shape != (expected,):
        raise MissingActionMask(
            f"{name} returned an action_mask of shape {mask.shape} from {where}, "
            f"expected ({expected},)"
        )
    return mask


def _update_device() -> torch.device:
    """``cuda`` when the accelerator is there, else CPU.

    HIP reports through ``torch.cuda.is_available()``, so this is the same
    check on a ROCm build as on a CUDA build.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _same_device(found: torch.device, target: torch.device) -> bool:
    """Compare unindexed ``cuda`` against the ``cuda:0`` a tensor reports."""
    return found.type == target.type and (target.index is None or found.index == target.index)


def _optimizer_moment_tensors(optimizer: Any) -> list[torch.Tensor]:
    """Adam ``exp_avg`` / ``exp_avg_sq``, not the 0-dim CPU ``step`` counter.

    PyTorch keeps ``state['step']`` on CPU. Treating that as a device miss
    made every post-train pin look unsettled, so the collect graph recaptured
    each update.
    """
    if optimizer is None:
        return []
    return [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    ]


def _pin_algorithm_device(model: Any, device: Any) -> None:
    """Move the one live policy, its Adam state, and the buffer to ``device``.

    ``nn.Module.to`` leaves optimizer state behind, and a stale Adam tensor
    raises at the next ``optimizer.step``. Idempotent, so a callback can call
    it every rollout. A 0-dim Adam ``step`` tensor stays on CPU.
    """
    target = torch.device(device)
    policy = model.policy
    optimizer = getattr(policy, "optimizer", None)
    buffer = getattr(model, "rollout_buffer", None)
    state_tensors = _optimizer_moment_tensors(optimizer)

    settled = (
        _same_device(model.device, target)
        and all(_same_device(p.device, target) for p in policy.parameters())
        and all(_same_device(t.device, target) for t in state_tensors)
        and (buffer is None or _same_device(buffer.device, target))
    )
    if settled:
        return

    before = _storage_ids(policy)
    policy.to(target)
    for state in optimizer.state.values() if optimizer is not None else ():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                state[key] = value.to(target)
    model.device = target
    if buffer is not None:
        buffer.device = target
    trunk = getattr(policy, "_collect_trunk", None)
    if trunk is not None and before != _storage_ids(policy):
        trunk.mark_stale()


class _HybridDeviceCallback(BaseCallback):
    """Pin the rollout device for collect and eval, the update device for train.

    ``MaskableEvalCallback`` runs inside ``collect_rollouts``, so it inherits
    the rollout device without knowing about any of this. The two devices
    match when an accelerator is present; the pin is then a no-op.
    """

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        _pin_algorithm_device(self.model, _rollout_device())

    def _on_rollout_start(self) -> None:
        _pin_algorithm_device(self.model, _rollout_device())

    def _on_rollout_end(self) -> None:
        _pin_algorithm_device(self.model, _update_device())

    def _on_training_end(self) -> None:
        _pin_algorithm_device(self.model, _rollout_device())


def _make_maskable_ppo(
    start: TrainStart,
    env: Any,
    hyperparams: TrainHyperparams,
    *,
    tensorboard_log: str,
    net_arch: dict[str, list[int]] | None,
) -> Any:
    """Build the one live algorithm on the rollout device.

    A loaded zip keeps the architecture it was saved with, so ``net_arch``
    reaches a fresh start only. ``load_maskable_checkpoint`` still loads on
    CPU, then the surface pin moves it.
    """
    if isinstance(start, FreshStart):
        from sb3_contrib import MaskablePPO

        from sts2_env.train.card_embed import (
            card_embed_policy_kwargs,
            is_run_observation_space,
        )

        if is_run_observation_space(env.observation_space):
            extra = card_embed_policy_kwargs(net_arch)
        elif net_arch is None:
            extra = {}
        else:
            extra = {"policy_kwargs": dict(net_arch=net_arch)}
        return MaskablePPO(
            "MlpPolicy",
            env,
            device=str(_rollout_device()),
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            verbose=1,
            tensorboard_log=tensorboard_log,
            **extra,
            **hyperparams.ppo_kwargs(),
        )

    return load_maskable_checkpoint(
        start.path,
        env,
        tensorboard_log=tensorboard_log,
        **hyperparams.ppo_kwargs(),
    )


@dataclass(frozen=True)
class _Parallelism:
    n_envs: int

    @staticmethod
    def parse(n_envs: int) -> _Parallelism:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        return _Parallelism(n_envs)

    def uses_subproc(self) -> bool:
        return self.n_envs > 1


@dataclass(frozen=True)
class _EvalSchedule:
    eval_freq: int
    n_eval_episodes: int

    @staticmethod
    def parse(eval_freq: int, n_eval_episodes: int) -> _EvalSchedule:
        if eval_freq < 1:
            raise ValueError(f"eval_freq must be >= 1, got {eval_freq}")
        if n_eval_episodes < 1:
            raise ValueError(f"n_eval_episodes must be >= 1, got {n_eval_episodes}")
        return _EvalSchedule(eval_freq, n_eval_episodes)

    def callback_eval_freq(self, parallelism: _Parallelism) -> int:
        return max(self.eval_freq // parallelism.n_envs, 1)


def _open_surface(
    factory: Callable[[], Any],
    parallelism: _Parallelism,
    schedule: _EvalSchedule,
    output_dir: Path,
    start: TrainStart,
    hyperparams: TrainHyperparams,
    *,
    net_arch: dict[str, list[int]] | None,
) -> MaskedTrainSurface:
    DummyVecEnv, SubprocVecEnv, MaskableEvalCallback = _require_sb3()
    torch.set_num_threads(TRAINER_TORCH_THREADS)

    factory = _CachedMaskFactory(factory)
    probe = factory()
    try:
        _reject_wrapped_env(probe)
        probe.reset()
        probe.action_masks()
    finally:
        probe.close()

    env_fns = [factory] * parallelism.n_envs
    if parallelism.uses_subproc():
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
    train_env = _ParentMaskVec(train_env)
    # The cache is empty until a reset, so read masks from the surface before learn.
    train_env.reset()
    eval_env = _ParentMaskVec(DummyVecEnv([factory]))

    output_dir = Path(output_dir)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=schedule.callback_eval_freq(parallelism),
        n_eval_episodes=schedule.n_eval_episodes,
        deterministic=False,
        use_masking=True,
    )
    model = _make_maskable_ppo(
        start,
        train_env,
        hyperparams,
        tensorboard_log=str(output_dir / "tb_logs"),
        net_arch=net_arch,
    )
    _pin_algorithm_device(model, _rollout_device())
    _install_collect_trunk(model, batch=parallelism.n_envs)
    return MaskedTrainSurface(train_env, eval_env, eval_callback, model)


def _reject_wrapped_env(env: Any) -> None:
    from sb3_contrib.common.wrappers import ActionMasker

    node: Any = env
    while node is not None:
        if isinstance(node, ActionMasker):
            raise TypeError(
                "probe env is wrapped in ActionMasker, which repeats the action_masks() "
                "the STS2 envs already expose"
            )
        node = node.env if isinstance(node, gymnasium.Wrapper) else None
    try:
        env.get_wrapper_attr("action_masks")
    except AttributeError as exc:
        raise TypeError(f"{type(env).__name__} does not define action_masks()") from exc


def _require_sb3():
    """Return the three SB3 types a job needs, or say what to install."""
    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise ImportError(
            "Training requires stable-baselines3 and sb3-contrib. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from exc
    return DummyVecEnv, SubprocVecEnv, MaskableEvalCallback
