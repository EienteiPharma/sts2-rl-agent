"""Hang-protocol combat transition buffer (offline collect → learn).

Schema, validation, IO, and constants for hang-protocol combat transition buffers.
Re-exports collect and replay modules for backwards compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

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
    hang_jev_flags,
)

BUFFER_VERSION = 1
REQUIRED_KEYS = ("obs", "next_obs", "action", "reward", "done", "action_mask")
HUNG_OUTDIR_NAME = "combat_ppo_obs_v1_bh_v1"
ONPOLICY_FROZEN_OUTDIR = "combat_runenv_onpolicy_v1"
ANTIFORGET_FROZEN_OUTDIR = "combat_runenv_antiforget_v1"
FROZEN_OUTDIR_NAMES = (
    HUNG_OUTDIR_NAME,
    ONPOLICY_FROZEN_OUTDIR,
    ANTIFORGET_FROZEN_OUTDIR,
)
PROTECTED_COLLECT_DIR_NAMES = (
    "runenv_combat_buffer_ep",
    "runenv_combat_buffer_colab_v1",
)
COLAB_V1_COLLECT_OUT = (
    "/workspace/sts2-sim/output/runenv_combat_buffer_colab_v1/transitions.npz"
)


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
    """Never write into bh_v1 / frozen onpolicy_v1 / frozen antiforget_v1 outdirs."""
    out = Path(path).expanduser()
    parts = set(out.parts)
    for name in FROZEN_OUTDIR_NAMES:
        if out.name == name or name in parts:
            raise SystemExit(
                f"refusing to overwrite frozen {what} {out}; pick a new outdir"
            )
    for name in PROTECTED_COLLECT_DIR_NAMES:
        if name in parts and out.name == "transitions.npz":
            hint = (
                "planning-only: keep --out elsewhere; set --planning-out default"
                if name == "runenv_combat_buffer_colab_v1"
                else f"use {COLAB_V1_COLLECT_OUT} for colab_v1"
            )
            raise SystemExit(
                f"refusing to overwrite protected collect {what} {out}; {hint}"
            )
    return out


def meta_json_path(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().with_suffix(".meta.json")


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


# Backwards compatibility re-exports from collect and replay modules
from sts2_env.gym_env.combat_collect import (
    CollectWorkerConfig,
    collect_parallel,
    collect_transitions,
    collect_worker,
    legal_random_action,
    split_worker_steps,
)
from sts2_env.gym_env.combat_replay import (
    CombatReplayEnv,
)

__all__ = [
    # Schema / IO
    "ANTIFORGET_FROZEN_OUTDIR",
    "BUFFER_VERSION",
    "FROZEN_OUTDIR_NAMES",
    "COLAB_V1_COLLECT_OUT",
    "HUNG_OUTDIR_NAME",
    "ONPOLICY_FROZEN_OUTDIR",
    "PROTECTED_COLLECT_DIR_NAMES",
    "REQUIRED_KEYS",
    "concat_buffers",
    "episode_starts",
    "hang_protocol_meta",
    "load_combat_buffer",
    "meta_json_path",
    "refuse_frozen_path",
    "save_combat_buffer",
    "stack_records",
    "synthetic_combat_buffer",
    "validate_buffer",
    # Collect (re-exported from combat_collect)
    "CollectWorkerConfig",
    "collect_parallel",
    "collect_transitions",
    "collect_worker",
    "legal_random_action",
    "split_worker_steps",
    # Replay (re-exported from combat_replay)
    "CombatReplayEnv",
]
