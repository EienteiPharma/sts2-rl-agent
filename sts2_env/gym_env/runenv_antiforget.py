"""Episode mixer: hang-protocol RunEnv combat + loadout_v1 fixture combat.

Same obs_v1 / action space as ``bh_v1``.  On each ``reset()`` pick the
RunEnv half with probability ``runenv_frac`` (default 0.30, loadout-dominant);
otherwise ``STS2CombatEnv`` with a ``loadout_v1`` provider.  PPO sees only
combat steps either way.

``--runenv-frac 0.7`` (antiforget_v1) is frozen: HOLD 50.0/77.8/22.2 FAIL.
Continue-from ``bh_v1`` only — refuse the antiforget_v1 zip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    OBS_VALUE_HIGH,
    OBS_VALUE_LOW,
    RunEnvOnPolicyCombatEnv,
)

DEFAULT_RUNENV_FRAC = 0.30
FROZEN_RUNENV_FRAC = 0.70
FROZEN_CONTINUE_FROM_MARKER = "antiforget_v1"
FROZEN_RUNENV_FRAC_NOTE = (
    "WARNING: --runenv-frac 0.7 is FROZEN (antiforget_v1 HOLD 50.0/77.8/22.2 FAIL). "
    "Recommended loadout-dominant mix is 0.3 from bh_v1."
)
SOURCE_RUNENV = "runenv"
SOURCE_LOADOUT = "loadout_v1"


def parse_runenv_frac(value: float | str) -> float:
    frac = float(value)
    if frac < 0.0 or frac > 1.0:
        raise SystemExit("--runenv-frac must be in [0, 1]")
    return frac


def frozen_runenv_frac_warning(frac: float) -> str | None:
    if abs(float(frac) - FROZEN_RUNENV_FRAC) < 1e-9:
        return FROZEN_RUNENV_FRAC_NOTE
    return None


def refuse_antiforget_v1_continue(path: str | Path) -> Path:
    """Never continue-from the frozen 0.7 antiforget_v1 zip. Use bh_v1."""
    zip_path = Path(path).expanduser()
    hay = str(zip_path).replace("\\", "/")
    if FROZEN_CONTINUE_FROM_MARKER in hay:
        raise SystemExit(
            f"refusing continue-from frozen antiforget_v1 zip {zip_path}; "
            "continue-from bh_v1 only (0.7 recipe is frozen; use loadout-dominant 0.3)"
        )
    return zip_path


def action_mask_fn(env):
    """Module-level mask fn so SubprocVecEnv / cloudpickle can find it."""
    return env.action_masks()


class MixedHangLoadoutEnvMaker:
    """Pickle-friendly env_fn for DummyVecEnv / SubprocVecEnv.

    Nested closures in ``scripts/`` fail stdlib pickle (spawn). This maker
    stores only ints/floats/optional buffer path. ``--n-envs > 1`` still
    hang-protocol Jev on a live RunEnv half; pass ``buffer_path`` to replay
    collected combat segments instead (TypeSafe off the learn path).
    """

    def __init__(
        self,
        seed: int = 0,
        runenv_frac: float = DEFAULT_RUNENV_FRAC,
        max_steps: int = 2000,
        buffer_path: str | None = None,
    ):
        self.seed = int(seed)
        self.runenv_frac = parse_runenv_frac(runenv_frac)
        self.max_steps = int(max_steps)
        self.buffer_path = buffer_path

    def __call__(self):
        runenv_env = None
        if self.buffer_path:
            from sts2_env.gym_env.combat_buffer import CombatReplayEnv

            runenv_env = CombatReplayEnv.from_path(self.buffer_path, seed=self.seed)
        return MixedHangLoadoutEnv(
            runenv_frac=self.runenv_frac,
            loadout_provider=make_loadout_v1_provider(offset=self.seed),
            max_steps=self.max_steps,
            seed_offset=self.seed,
            runenv_env=runenv_env,
            jev_key_index=self.seed,
        )


class MaskedEnvMaker:
    """Wrap a pickleable env maker with sb3 ActionMasker (optional)."""

    def __init__(self, inner: Callable, ActionMasker: Any = None):
        self.inner = inner
        self.ActionMasker = ActionMasker

    def __call__(self):
        env = self.inner()
        if self.ActionMasker is not None:
            env = self.ActionMasker(env, action_mask_fn)
        return env


class MixedHangLoadoutEnv(gymnasium.Env):
    """One Gym env; episode source is sampled on reset."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        runenv_frac: float = DEFAULT_RUNENV_FRAC,
        loadout_provider: Callable[[], dict[str, Any]] | None = None,
        max_steps: int = 2000,
        seed_offset: int = 0,
        jev_adapter: Any | None = None,
        render_mode: str | None = None,
        runenv_env: gymnasium.Env | None = None,
        jev_key_index: int = 0,
    ):
        super().__init__()
        self.runenv_frac = parse_runenv_frac(runenv_frac)
        if loadout_provider is None:
            raise SystemExit("MixedHangLoadoutEnv requires a loadout_v1 provider")
        self.observation_space = spaces.Box(
            low=OBS_VALUE_LOW,
            high=OBS_VALUE_HIGH,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._runenv = runenv_env or RunEnvOnPolicyCombatEnv(
            max_steps=max_steps,
            seed_offset=seed_offset,
            jev_adapter=jev_adapter,
            jev_key_index=jev_key_index,
            render_mode=render_mode,
        )
        self._loadout = STS2CombatEnv(loadout_provider=loadout_provider)
        self._active: gymnasium.Env | None = None
        self._source = SOURCE_RUNENV
        self._rng = np.random.RandomState(0)

    def hang_protocol(self) -> dict[str, Any]:
        proto = self._runenv.hang_protocol()
        proto["runenv_frac"] = self.runenv_frac
        proto["loadout_frac"] = round(1.0 - self.runenv_frac, 4)
        proto["loadout_half"] = SOURCE_LOADOUT
        return proto

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**31 - 1)
        )
        self._rng = np.random.RandomState(rng_seed)
        if self._rng.random() < self.runenv_frac:
            self._active = self._runenv
            self._source = SOURCE_RUNENV
        else:
            self._active = self._loadout
            self._source = SOURCE_LOADOUT
        obs, info = self._active.reset(seed=seed, options=options)
        info = dict(info or {})
        info["mix_source"] = self._source
        info["runenv_frac"] = self.runenv_frac
        if self._source == SOURCE_RUNENV:
            info.setdefault("start_with_neow", HANG_START_WITH_NEOW)
            info.setdefault("jev_event", HANG_JEV_EVENT)
            info.setdefault("jev_neow", HANG_JEV_NEOW)
        else:
            info["loadout"] = SOURCE_LOADOUT
            info.setdefault("start_with_neow", False)
        return obs, info

    def step(self, action: int):
        if self._active is None:
            raise RuntimeError("reset() before step()")
        obs, reward, terminated, truncated, info = self._active.step(action)
        info = dict(info or {})
        info["mix_source"] = self._source
        info["runenv_frac"] = self.runenv_frac
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._active is None:
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
            mask[0] = 1
            return mask
        return self._active.action_masks()

    def close(self):
        self._runenv.close()
        self._loadout.close()

    def render(self):
        if self._active is not None:
            return self._active.render()
        return None


def make_loadout_v1_provider(offset: int = 0):
    """Rotate PR ``loadout_v1`` train fixtures (not mix_neow_v1)."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_combat.py"
    spec = importlib.util.spec_from_file_location("train_combat_antiforget", path)
    assert spec is not None and spec.loader is not None
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod.make_loadout_provider("loadout_v1", offset=offset)
