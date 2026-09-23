"""Hang-protocol combat transition buffer (offline collect → learn).

Collector rolls ``RunEnvOnPolicyCombatEnv`` (MAP/REST/CARD Jev still on).
Only combat transitions are stored. ``CombatReplayEnv`` replays them so
``MaskablePPO.learn`` never calls TypeSafe.

Keys: obs, next_obs, action, reward, done, action_mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_ASCENSION,
    HANG_CHARACTER,
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_JEV_PHASES,
    HANG_START_WITH_NEOW,
    OBS_VALUE_HIGH,
    OBS_VALUE_LOW,
    hang_jev_flags,
)

BUFFER_VERSION = 1
REQUIRED_KEYS = ("obs", "next_obs", "action", "reward", "done", "action_mask")
HUNG_OUTDIR_NAME = "combat_ppo_obs_v1_bh_v1"
ONPOLICY_FROZEN_OUTDIR = "combat_runenv_onpolicy_v1"
FROZEN_OUTDIR_NAMES = (HUNG_OUTDIR_NAME, ONPOLICY_FROZEN_OUTDIR)


def hang_protocol_meta() -> dict[str, Any]:
    flags = hang_jev_flags()
    return {
        "jev": HANG_JEV,
        "jev_event": HANG_JEV_EVENT,
        "jev_neow": HANG_JEV_NEOW,
        "jev_phases": HANG_JEV_PHASES,
        "start_with_neow": HANG_START_WITH_NEOW,
        "character": HANG_CHARACTER,
        "ascension": HANG_ASCENSION,
        "allows_event": flags.allows_event(),
        "allows_neow": flags.allows_neow(),
        "combat_obs_size": OBS_SIZE,
        "combat_action_size": ACTION_SPACE_SIZE,
    }


def refuse_frozen_path(path: str | Path, *, what: str = "path") -> Path:
    """Never write into bh_v1 / frozen onpolicy_v1 outdirs."""
    out = Path(path).expanduser()
    parts = set(out.parts)
    for name in FROZEN_OUTDIR_NAMES:
        if out.name == name or name in parts:
            raise SystemExit(
                f"refusing to overwrite frozen {what} {out}; pick a new outdir"
            )
    return out


def meta_json_path(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().with_suffix(".meta.json")


def split_worker_steps(n_steps: int, n_envs: int) -> list[int]:
    n_envs = max(1, int(n_envs))
    n_steps = max(0, int(n_steps))
    if n_steps == 0:
        return [0] * n_envs
    base, rem = divmod(n_steps, n_envs)
    return [base + (1 if i < rem else 0) for i in range(n_envs)]


def legal_random_action(mask, rng: np.random.RandomState) -> int:
    valid = np.flatnonzero(np.asarray(mask) == 1)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def validate_buffer(arrays: dict[str, np.ndarray]) -> int:
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"combat buffer missing keys: {missing}")
    n = int(np.asarray(arrays["obs"]).shape[0])
    if n == 0:
        raise ValueError("combat buffer is empty")
    obs = np.asarray(arrays["obs"])
    next_obs = np.asarray(arrays["next_obs"])
    action = np.asarray(arrays["action"])
    reward = np.asarray(arrays["reward"])
    done = np.asarray(arrays["done"])
    mask = np.asarray(arrays["action_mask"])
    if obs.ndim != 2 or obs.shape != (n, OBS_SIZE):
        raise ValueError(f"obs shape {obs.shape} != ({n}, {OBS_SIZE})")
    if next_obs.shape != (n, OBS_SIZE):
        raise ValueError(f"next_obs shape {next_obs.shape} != ({n}, {OBS_SIZE})")
    if action.shape != (n,):
        raise ValueError(f"action shape {action.shape} != ({n},)")
    if reward.shape != (n,):
        raise ValueError(f"reward shape {reward.shape} != ({n},)")
    if done.shape != (n,):
        raise ValueError(f"done shape {done.shape} != ({n},)")
    if mask.shape != (n, ACTION_SPACE_SIZE):
        raise ValueError(
            f"action_mask shape {mask.shape} != ({n}, {ACTION_SPACE_SIZE})"
        )
    return n


def stack_records(records: dict[str, list]) -> dict[str, np.ndarray]:
    if not records["obs"]:
        raise ValueError("no combat transitions collected")
    arrays = {
        "obs": np.asarray(records["obs"], dtype=np.float32),
        "next_obs": np.asarray(records["next_obs"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.int64),
        "reward": np.asarray(records["reward"], dtype=np.float32),
        "done": np.asarray(records["done"], dtype=np.bool_),
        "action_mask": np.asarray(records["action_mask"], dtype=np.int8),
    }
    validate_buffer(arrays)
    return arrays


def concat_buffers(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no buffer shards to concat")
    for part in parts:
        validate_buffer(part)
    arrays = {
        key: np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
        for key in REQUIRED_KEYS
    }
    validate_buffer(arrays)
    return arrays


def save_combat_buffer(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> Path:
    out = refuse_frozen_path(path, what="buffer")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = validate_buffer(arrays)
    payload = hang_protocol_meta()
    payload.update(meta or {})
    payload["buffer_version"] = BUFFER_VERSION
    payload["n_transitions"] = n
    payload["obs_size"] = OBS_SIZE
    payload["action_size"] = ACTION_SPACE_SIZE
    payload["keys"] = list(REQUIRED_KEYS)
    np.savez_compressed(out, **{k: arrays[k] for k in REQUIRED_KEYS})
    meta_json_path(out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return out


def load_combat_buffer(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = Path(path).expanduser()
    if not npz_path.is_file():
        raise SystemExit(f"combat buffer not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as blob:
        arrays = {key: np.asarray(blob[key]) for key in REQUIRED_KEYS}
    validate_buffer(arrays)
    meta: dict[str, Any] = {}
    mpath = meta_json_path(npz_path)
    if mpath.is_file():
        meta = json.loads(mpath.read_text())
    return arrays, meta


def synthetic_combat_buffer(
    n: int = 16,
    *,
    n_episodes: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Deterministic fake hang-combat buffer for dry-run / unit tests."""
    rng = np.random.RandomState(seed)
    n = max(1, int(n))
    n_episodes = max(1, min(int(n_episodes), n))
    obs = rng.randn(n, OBS_SIZE).astype(np.float32)
    next_obs = rng.randn(n, OBS_SIZE).astype(np.float32)
    action = rng.randint(0, ACTION_SPACE_SIZE, size=n, dtype=np.int64)
    reward = rng.randn(n).astype(np.float32)
    done = np.zeros(n, dtype=np.bool_)
    bounds = np.linspace(0, n, n_episodes + 1, dtype=int)
    for end in bounds[1:]:
        done[end - 1] = True
    next_obs[:-1] = obs[1:]
    mask = np.zeros((n, ACTION_SPACE_SIZE), dtype=np.int8)
    mask[:, 0] = 1
    for i, act in enumerate(action):
        mask[i, int(act)] = 1
    arrays = {
        "obs": obs,
        "next_obs": next_obs,
        "action": action,
        "reward": reward,
        "done": done,
        "action_mask": mask,
    }
    validate_buffer(arrays)
    return arrays


def episode_starts(done: np.ndarray) -> np.ndarray:
    flags = np.asarray(done, dtype=np.bool_)
    if flags.size == 0:
        return np.zeros(0, dtype=np.int64)
    starts = [0]
    for i, flag in enumerate(flags[:-1]):
        if flag:
            starts.append(i + 1)
    return np.asarray(starts, dtype=np.int64)


def collect_transitions(
    env,
    n_steps: int,
    *,
    rng: np.random.RandomState,
    select_fn,
    reset_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Roll a hang-protocol combat env; store combat steps only."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    seed = int(reset_seed) if reset_seed is not None else int(rng.randint(0, 2**31 - 1))
    obs, info = env.reset(seed=seed)
    records: dict[str, list] = {k: [] for k in REQUIRED_KEYS}
    empty = 0
    steps = 0
    while steps < n_steps:
        if info.get("combat_unreachable"):
            empty += 1
            if empty > 8:
                break
            seed = int(rng.randint(0, 2**31 - 1))
            obs, info = env.reset(seed=seed)
            continue
        mask = env.action_masks()
        action = int(select_fn(obs, mask))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        records["obs"].append(np.asarray(obs, dtype=np.float32))
        records["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        records["action"].append(action)
        records["reward"].append(float(reward))
        records["done"].append(done)
        records["action_mask"].append(np.asarray(mask, dtype=np.int8))
        steps += 1
        if done:
            seed = int(rng.randint(0, 2**31 - 1))
            obs, info = env.reset(seed=seed)
        else:
            obs = next_obs
    return stack_records(records)


def collect_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level multiprocessing target. Each worker is hang-protocol Jev."""
    from sts2_env.eval.jev import load_typesafe_api_keys
    from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

    n_steps = int(payload["n_steps"])
    worker_id = int(payload.get("worker_id", 0))
    seed = int(payload.get("seed", 0))
    policy = str(payload.get("policy", "random"))
    shard = Path(payload["shard"])
    max_steps = int(payload.get("max_steps", 2000))
    rng = np.random.RandomState(seed + worker_id * 100003)
    load_typesafe_api_keys()

    env = RunEnvOnPolicyCombatEnv(
        max_steps=max_steps,
        seed_offset=seed + worker_id,
        jev_key_index=worker_id,
    )
    model = None
    if policy == "model":
        model_path = payload.get("model")
        if not model_path or not Path(model_path).is_file():
            env.close()
            raise SystemExit(f"collect --policy model zip not found: {model_path}")
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as e:
            env.close()
            raise SystemExit(
                "collect --policy model requires sb3-contrib / torch"
            ) from e
        model = MaskablePPO.load(str(model_path), device="cpu")

    def select_fn(obs, mask):
        if model is None:
            return legal_random_action(mask, rng)
        action, _ = model.predict(
            np.asarray(obs), action_masks=mask, deterministic=False
        )
        return int(action)

    try:
        arrays = collect_transitions(
            env,
            n_steps,
            rng=rng,
            select_fn=select_fn,
            reset_seed=seed + worker_id,
        )
    finally:
        env.close()

    meta = hang_protocol_meta()
    meta.update(
        {
            "worker_id": worker_id,
            "policy": policy,
            "n_steps_requested": n_steps,
            "seed": seed,
        }
    )
    save_combat_buffer(shard, arrays, meta)
    return {
        "shard": str(shard),
        "n_transitions": int(arrays["obs"].shape[0]),
        "worker_id": worker_id,
    }


def collect_parallel(
    *,
    out_path: str | Path,
    n_steps: int,
    n_envs: int = 1,
    policy: str = "random",
    model: str | None = None,
    seed: int = 0,
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Collect hang combat transitions, optionally across ``n_envs`` workers."""
    from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs

    out = refuse_frozen_path(out_path, what="buffer")
    n_envs = max(1, int(n_envs))
    load_typesafe_api_keys()
    note = warn_n_envs(n_envs)
    if note:
        print(note)
    quotas = [q for q in split_worker_steps(n_steps, n_envs) if q > 0]
    shard_dir = out.parent / f".{out.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for i, quota in enumerate(quotas):
        payloads.append(
            {
                "n_steps": quota,
                "worker_id": i,
                "seed": int(seed),
                "policy": policy,
                "model": model,
                "shard": str(shard_dir / f"shard_{i:02d}.npz"),
                "max_steps": int(max_steps),
            }
        )
    if len(payloads) == 1:
        results = [collect_worker(payloads[0])]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(collect_worker, payloads)
    parts = []
    for row in results:
        arrays, _meta = load_combat_buffer(row["shard"])
        parts.append(arrays)
    merged = concat_buffers(parts)
    meta = hang_protocol_meta()
    meta.update(
        {
            "policy": policy,
            "n_envs": len(payloads),
            "n_steps_requested": int(n_steps),
            "seed": int(seed),
            "shards": [r["shard"] for r in results],
            **typesafe_key_pool_summary(),
        }
    )
    save_combat_buffer(out, merged, meta)
    return {
        "out": str(out),
        "n_transitions": int(merged["obs"].shape[0]),
        "n_envs": len(payloads),
        "policy": policy,
        "hang": hang_protocol_meta(),
        **typesafe_key_pool_summary(),
    }


class CombatReplayEnv(gymnasium.Env):
    """Replay stored hang-protocol combat transitions (no Jev / TypeSafe).

    ``step(action)`` advances the stored sequence. The collector action is
    kept for offline diagnostics; PPO's sampled action does not re-roll
    RunEnv (that would put Jev back on the learn path).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        meta: dict[str, Any] | None = None,
        *,
        seed: int = 0,
    ):
        super().__init__()
        validate_buffer(arrays)
        self._obs = np.asarray(arrays["obs"], dtype=np.float32)
        self._next_obs = np.asarray(arrays["next_obs"], dtype=np.float32)
        self._action = np.asarray(arrays["action"], dtype=np.int64)
        self._reward = np.asarray(arrays["reward"], dtype=np.float32)
        self._done = np.asarray(arrays["done"], dtype=np.bool_)
        self._mask = np.asarray(arrays["action_mask"], dtype=np.int8)
        self._n = int(self._obs.shape[0])
        self._starts = episode_starts(self._done)
        self._meta = dict(hang_protocol_meta())
        self._meta.update(meta or {})
        self.observation_space = spaces.Box(
            low=OBS_VALUE_LOW,
            high=OBS_VALUE_HIGH,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._cursor = 0
        self._ep_i = -1
        self._rng = np.random.RandomState(int(seed))

    @classmethod
    def from_path(cls, path: str | Path, *, seed: int = 0) -> "CombatReplayEnv":
        arrays, meta = load_combat_buffer(path)
        return cls(arrays, meta, seed=seed)

    def hang_protocol(self) -> dict[str, Any]:
        proto = dict(self._meta)
        proto.setdefault("replay", True)
        proto.setdefault("n_transitions", self._n)
        return proto

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
            self._cursor = int(self._starts[int(self._rng.randint(0, len(self._starts)))])
            self._ep_i = 0
        else:
            self._ep_i = (self._ep_i + 1) % len(self._starts)
            self._cursor = int(self._starts[self._ep_i])
        return self._obs[self._cursor], self._info(reset=True)

    def step(self, action: int):
        i = int(self._cursor)
        next_obs = self._next_obs[i]
        reward = float(self._reward[i])
        terminated = bool(self._done[i])
        truncated = False
        if i + 1 >= self._n:
            truncated = not terminated
            terminated = True
        else:
            self._cursor = i + 1
        if terminated or truncated:
            self._cursor = i
        info = self._info(reset=False)
        info["stored_action"] = int(self._action[i])
        info["step_action"] = int(action)
        return next_obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._mask[self._cursor]

    def close(self):
        return None

    def _info(self, *, reset: bool) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks(),
            "replay": True,
            "buffer_index": int(self._cursor),
            "mix_source": "runenv_buffer",
            "start_with_neow": HANG_START_WITH_NEOW,
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "loadout": None,
            "combat_obs_size": OBS_SIZE,
            "reset": reset,
        }
