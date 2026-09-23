"""Hang-protocol RunEnv combat + loadout_v1 fixture mixer.

Same obs_v1 / action space as ``bh_v1``.  PPO sees only combat steps.

``mix_by="episodes"`` (live online default): on each ``reset()`` pick the
RunEnv half with probability ``runenv_frac``.  Hang buffer episodes are
~50× longer than loadout fixture fights, so this is **not** a step mix —
``frac=0.3`` was ~94% buffer timesteps (loadoutdom_ep_v1 HOLD collapse).

``mix_by="steps"`` (from_buffer / CombatReplayEnv default): on reset, pick
the source that drives **cumulative step fraction** toward ``runenv_frac``
(under-represented source wins, with jitter).

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
MIX_BY_STEPS = "steps"
MIX_BY_EPISODES = "episodes"
MIX_BY_CHOICES = (MIX_BY_STEPS, MIX_BY_EPISODES)
DEFAULT_MIX_BY_BUFFER = MIX_BY_STEPS
DEFAULT_MIX_BY_LIVE = MIX_BY_EPISODES
MIX_BY_STEPS_JITTER = 0.05


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


def parse_mix_by(value: str) -> str:
    s = str(value).strip().lower()
    if s not in MIX_BY_CHOICES:
        raise SystemExit("--mix-by must be 'steps' or 'episodes'")
    return s


def is_buffer_replay_env(env: Any) -> bool:
    if env is None:
        return False
    if type(env).__name__ == "CombatReplayEnv":
        return True
    proto = getattr(env, "hang_protocol", None)
    if callable(proto):
        try:
            return bool(proto().get("replay"))
        except Exception:
            return False
    return False


def resolve_mix_by(mix_by: str | None, *, buffer: bool) -> str:
    if mix_by is None or str(mix_by).strip() == "" or str(mix_by).strip().lower() == "auto":
        return DEFAULT_MIX_BY_BUFFER if buffer else DEFAULT_MIX_BY_LIVE
    return parse_mix_by(mix_by)


def select_mix_source(
    *,
    mix_by: str,
    runenv_frac: float,
    n_runenv_steps: int,
    n_loadout_steps: int,
    rng: np.random.RandomState,
    jitter: float = MIX_BY_STEPS_JITTER,
) -> str:
    """Pick episode source. ``steps`` targets cumulative step fraction."""
    frac = parse_runenv_frac(runenv_frac)
    if frac <= 0.0:
        return SOURCE_LOADOUT
    if frac >= 1.0:
        return SOURCE_RUNENV
    mode = parse_mix_by(mix_by)
    if mode == MIX_BY_EPISODES:
        return SOURCE_RUNENV if rng.random() < frac else SOURCE_LOADOUT
    total = int(n_runenv_steps) + int(n_loadout_steps)
    current = (float(n_runenv_steps) / total) if total > 0 else 0.0
    noise = 0.0
    if jitter and total > 0:
        noise = float(jitter) * (2.0 * float(rng.random()) - 1.0)
    if current + noise < frac:
        return SOURCE_RUNENV
    return SOURCE_LOADOUT


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
        mix_by: str | None = None,
    ):
        self.seed = int(seed)
        self.runenv_frac = parse_runenv_frac(runenv_frac)
        self.max_steps = int(max_steps)
        self.buffer_path = buffer_path
        self.mix_by = mix_by

    def __call__(self):
        runenv_env = None
        if self.buffer_path:
            from sts2_env.gym_env.combat_replay import CombatReplayEnv

            runenv_env = CombatReplayEnv.from_path(self.buffer_path, seed=self.seed)
        return MixedHangLoadoutEnv(
            runenv_frac=self.runenv_frac,
            loadout_provider=make_loadout_v1_provider(offset=self.seed),
            max_steps=self.max_steps,
            seed_offset=self.seed,
            runenv_env=runenv_env,
            jev_key_index=self.seed,
            mix_by=self.mix_by,
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
    """One Gym env; episode source is chosen on reset (episodes or steps)."""

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
        loadout_env: gymnasium.Env | None = None,
        jev_key_index: int = 0,
        mix_by: str | None = None,
    ):
        super().__init__()
        self.runenv_frac = parse_runenv_frac(runenv_frac)
        if loadout_env is None and loadout_provider is None:
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
        self._loadout = loadout_env or STS2CombatEnv(loadout_provider=loadout_provider)
        self.mix_by = resolve_mix_by(
            mix_by, buffer=is_buffer_replay_env(self._runenv)
        )
        self._active: gymnasium.Env | None = None
        self._source = SOURCE_RUNENV
        self._rng = np.random.RandomState(0)
        self._n_runenv_steps = 0
        self._n_loadout_steps = 0
        self._n_runenv_eps = 0
        self._n_loadout_eps = 0

    def hang_protocol(self) -> dict[str, Any]:
        proto = self._runenv.hang_protocol()
        proto["runenv_frac"] = self.runenv_frac
        proto["loadout_frac"] = round(1.0 - self.runenv_frac, 4)
        proto["loadout_half"] = SOURCE_LOADOUT
        proto["mix_by"] = self.mix_by
        proto["mix_steps_runenv"] = int(self._n_runenv_steps)
        proto["mix_steps_loadout"] = int(self._n_loadout_steps)
        total = self._n_runenv_steps + self._n_loadout_steps
        proto["mix_step_frac_runenv"] = (
            (self._n_runenv_steps / total) if total else None
        )
        return proto

    def _mix_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        out = dict(info or {})
        total = self._n_runenv_steps + self._n_loadout_steps
        out["mix_source"] = self._source
        out["runenv_frac"] = self.runenv_frac
        out["mix_by"] = self.mix_by
        out["mix_steps_runenv"] = int(self._n_runenv_steps)
        out["mix_steps_loadout"] = int(self._n_loadout_steps)
        out["mix_step_frac_runenv"] = (
            (self._n_runenv_steps / total) if total else None
        )
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**31 - 1)
        )
        self._rng = np.random.RandomState(rng_seed)
        self._source = select_mix_source(
            mix_by=self.mix_by,
            runenv_frac=self.runenv_frac,
            n_runenv_steps=self._n_runenv_steps,
            n_loadout_steps=self._n_loadout_steps,
            rng=self._rng,
        )
        if self._source == SOURCE_RUNENV:
            self._active = self._runenv
            self._n_runenv_eps += 1
        else:
            self._active = self._loadout
            self._n_loadout_eps += 1
        obs, info = self._active.reset(seed=seed, options=options)
        info = self._mix_info(info)
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
        if self._source == SOURCE_RUNENV:
            self._n_runenv_steps += 1
        else:
            self._n_loadout_steps += 1
        info = self._mix_info(info)
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


class FixedLengthCombatEnv(gymnasium.Env):
    """Deterministic combat stub for mix_by probes. Not a hang env."""

    metadata = {"render_modes": []}

    def __init__(self, n_steps: int = 5, *, source: str = SOURCE_LOADOUT):
        super().__init__()
        self.n_steps = max(1, int(n_steps))
        self.source = source
        self.observation_space = spaces.Box(
            low=OBS_VALUE_LOW,
            high=OBS_VALUE_HIGH,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self._t = 0
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        self._mask[0] = 1

    def hang_protocol(self) -> dict[str, Any]:
        return {"replay": self.source == SOURCE_RUNENV, "stub": True}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs.copy(), {
            "action_mask": self._mask.copy(),
            "mix_source": self.source,
        }

    def step(self, action: int):
        self._t += 1
        done = self._t >= self.n_steps
        return (
            self._obs.copy(),
            0.0,
            done,
            False,
            {"action_mask": self._mask.copy(), "mix_source": self.source},
        )

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def close(self):
        return None


def probe_mix_step_fraction(
    *,
    runenv_frac: float = DEFAULT_RUNENV_FRAC,
    mix_by: str = MIX_BY_STEPS,
    runenv_ep_len: int = 200,
    loadout_ep_len: int = 5,
    min_steps: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Roll stub mix until ``min_steps``; report episode vs step fractions."""
    env = MixedHangLoadoutEnv(
        runenv_frac=runenv_frac,
        loadout_env=FixedLengthCombatEnv(loadout_ep_len, source=SOURCE_LOADOUT),
        runenv_env=FixedLengthCombatEnv(runenv_ep_len, source=SOURCE_RUNENV),
        mix_by=mix_by,
    )
    obs, info = env.reset(seed=seed)
    n_eps = 1
    while (env._n_runenv_steps + env._n_loadout_steps) < int(min_steps):
        mask = env.action_masks()
        action = int(np.flatnonzero(np.asarray(mask) == 1)[0])
        _obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if (env._n_runenv_steps + env._n_loadout_steps) >= int(min_steps):
                break
            obs, info = env.reset()
            n_eps += 1
    steps_runenv = int(env._n_runenv_steps)
    steps_loadout = int(env._n_loadout_steps)
    eps_runenv = int(env._n_runenv_eps)
    eps_loadout = int(env._n_loadout_eps)
    env.close()
    total_steps = steps_runenv + steps_loadout
    total_eps = eps_runenv + eps_loadout
    step_frac = steps_runenv / total_steps if total_steps else 0.0
    ep_frac = eps_runenv / total_eps if total_eps else 0.0
    return {
        "mix_by": mix_by,
        "runenv_frac": runenv_frac,
        "runenv_ep_len": int(runenv_ep_len),
        "loadout_ep_len": int(loadout_ep_len),
        "n_steps": int(total_steps),
        "n_episodes": int(total_eps),
        "n_reset": int(n_eps),
        "steps_runenv": steps_runenv,
        "steps_loadout": steps_loadout,
        "eps_runenv": eps_runenv,
        "eps_loadout": eps_loadout,
        "step_runenv": step_frac,
        "ep_runenv": ep_frac,
        "mean_steps_runenv": runenv_ep_len,
        "mean_steps_loadout": loadout_ep_len,
    }
