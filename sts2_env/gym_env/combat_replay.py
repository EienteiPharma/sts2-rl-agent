"""Replay stored hang-protocol combat transitions (no Jev / TypeSafe).

Provides ``CombatReplayEnv`` which replays stored transitions so
``MaskablePPO.learn`` never calls TypeSafe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.combat_buffer import (
    episode_starts,
    hang_protocol_meta,
    load_combat_buffer,
    validate_buffer,
)
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    OBS_VALUE_HIGH,
    OBS_VALUE_LOW,
)


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


__all__ = [
    "CombatReplayEnv",
]
