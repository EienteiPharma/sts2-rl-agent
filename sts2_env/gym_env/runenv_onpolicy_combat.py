"""Hang-protocol RunEnv wrapper that exposes combat-only PPO steps.

``MaskablePPO`` continues a combat zip (obs_v1 / ``OBS_SIZE``=181,
``ACTION_SPACE_SIZE``).  ``STS2RunEnv`` rolls the hang protocol
(``--start-with-neow``, MAP/REST/CARD Jev, EVENT off, Neow random).
Non-combat phases are auto-stepped with the same
``choose_jev_noncombat`` path as ``scripts/eval_act1_runenv.py`` and never
enter the PPO rollout (zero gradient).

This is **not** ``STS2CombatEnv`` loadout-fixture training and **not**
``train_full_run.py`` (full RunEnv action space).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.eval.jev import build_jev_adapter
from sts2_env.eval.jev_policy import (
    JevPolicyFlags,
    choose_jev_noncombat,
    resolve_jev_flags,
)
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.reward import compute_reward
from sts2_env.gym_env.run_env import (
    STS2RunEnv,
    _COMBAT_SIZE,
    _COMBAT_START,
)
from sts2_env.run.run_manager import RunManager

# Locked hang protocol (do not retune here). Matches docs/HANG_PROTOCOL_2026-09-22.md.
HANG_JEV = "on"
HANG_JEV_EVENT = "off"
HANG_JEV_NEOW = "off"
HANG_JEV_PHASES = "map,rest,card"
HANG_START_WITH_NEOW = True
HANG_CHARACTER = "Ironclad"
HANG_ASCENSION = 0

DEFAULT_MAX_AUTO_STEPS = 2_000
DEFAULT_MAX_RESET_RETRIES = 16
OBS_VALUE_LOW = -1.0
OBS_VALUE_HIGH = 10.0


def hang_jev_flags() -> JevPolicyFlags:
    """MAP/REST/CARD Jev on; EVENT off; Neow random (hang default)."""
    return resolve_jev_flags(
        jev_event=HANG_JEV_EVENT,
        jev_phases=HANG_JEV_PHASES,
        jev_neow=HANG_JEV_NEOW,
    )


def is_combat_phase(env: STS2RunEnv) -> bool:
    mgr = getattr(env, "_mgr", None)
    return mgr is not None and mgr.phase == RunManager.PHASE_COMBAT and not mgr.is_over


def _run_over(env: STS2RunEnv) -> bool:
    mgr = getattr(env, "_mgr", None)
    return mgr is None or mgr.is_over


def _selected_combat_owner(mgr: RunManager, combat):
    actions = mgr.get_available_actions()
    selected_action = next(
        (a for a in actions if a.get("action") == "select_player" and a.get("selected")),
        None,
    )
    selected_owner = combat.primary_player
    if selected_action is not None:
        for state in combat.combat_player_states:
            if state.player_state.player_id == selected_action.get("player_id"):
                selected_owner = state.creature
                break
    return selected_owner


def combat_observation(env: STS2RunEnv) -> np.ndarray:
    """Combat obs_v1 for the hung zip. Never returns RunEnv obs."""
    mgr = getattr(env, "_mgr", None)
    if mgr is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)
    combat = mgr.get_combat_state()
    if combat is None:
        return np.zeros(OBS_SIZE, dtype=np.float32)
    obs = encode_observation(combat)
    if obs.shape[-1] != OBS_SIZE:
        raise RuntimeError(
            f"combat obs width {obs.shape[-1]} != OBS_SIZE={OBS_SIZE}"
        )
    return obs


def combat_action_mask(env: STS2RunEnv) -> np.ndarray:
    mgr = getattr(env, "_mgr", None)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    if mgr is None or mgr.phase != RunManager.PHASE_COMBAT:
        mask[0] = 1
        return mask
    combat = mgr.get_combat_state()
    if combat is None:
        mask[0] = 1
        return mask
    owner = _selected_combat_owner(mgr, combat)
    raw = get_action_mask(combat, owner=owner)
    n = min(len(raw), ACTION_SPACE_SIZE)
    mask[:n] = raw[:n]
    if int(mask.sum()) == 0:
        mask[0] = 1
    return mask


def to_runenv_combat_action(local: int) -> int:
    """Map a combat-zip action index into the RunEnv combat slice."""
    local = int(local)
    local = max(0, min(local, _COMBAT_SIZE - 1, ACTION_SPACE_SIZE - 1))
    return _COMBAT_START + local


class RunEnvOnPolicyCombatEnv(gymnasium.Env):
    """Gymnasium env: PPO timestep = one hang-protocol combat step.

    Episode is one STS2 run.  Between combats the wrapper auto-steps
    non-combat with hang hierarchical+Jev (frozen; no PPO transition).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        max_steps: int = DEFAULT_MAX_AUTO_STEPS,
        max_combat_turns: int = 200,
        max_auto_steps: int = DEFAULT_MAX_AUTO_STEPS,
        max_reset_retries: int = DEFAULT_MAX_RESET_RETRIES,
        jev_adapter: Any | None = None,
        jev_flags: JevPolicyFlags | None = None,
        render_mode: str | None = None,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=OBS_VALUE_LOW,
            high=OBS_VALUE_HIGH,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.max_auto_steps = int(max_auto_steps)
        self.max_reset_retries = int(max_reset_retries)
        self.seed_offset = int(seed_offset)
        self.render_mode = render_mode
        self.jev_flags = jev_flags or hang_jev_flags()
        self.jev_adapter = jev_adapter or build_jev_adapter(enabled=True)
        self.inner = STS2RunEnv(
            character_id=HANG_CHARACTER,
            ascension_level=HANG_ASCENSION,
            max_steps=max_steps,
            max_combat_turns=max_combat_turns,
            render_mode=render_mode,
        )
        self._rng = np.random.RandomState(0)
        self._combat_steps = 0
        self._noncombat_auto_steps = 0
        self._last_obs = np.zeros(OBS_SIZE, dtype=np.float32)

    def hang_protocol(self) -> dict[str, Any]:
        return {
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "jev_phases": HANG_JEV_PHASES,
            "start_with_neow": HANG_START_WITH_NEOW,
            "character": HANG_CHARACTER,
            "ascension": HANG_ASCENSION,
            "allows_event": self.jev_flags.allows_event(),
            "allows_neow": self.jev_flags.allows_neow(),
            "combat_obs_size": OBS_SIZE,
            "combat_action_size": ACTION_SPACE_SIZE,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2**31 - 1)
        )
        self._rng = np.random.RandomState(rng_seed)
        self._combat_steps = 0
        self._noncombat_auto_steps = 0

        last_info: dict[str, Any] = {}
        for attempt in range(self.max_reset_retries):
            inner_seed = rng_seed + self.seed_offset + attempt
            _, last_info = self.inner.reset(
                seed=inner_seed,
                options={"start_with_neow": HANG_START_WITH_NEOW},
            )
            done, _trunc = self._auto_noncombat()
            if is_combat_phase(self.inner):
                self._last_obs = combat_observation(self.inner)
                return self._last_obs, self._build_info(last_info)
            if done:
                continue
        self._last_obs = np.zeros(OBS_SIZE, dtype=np.float32)
        info = self._build_info(last_info)
        info["combat_unreachable"] = True
        return self._last_obs, info

    def step(self, action: int):
        if not is_combat_phase(self.inner):
            done, truncated = self._auto_noncombat()
            if not is_combat_phase(self.inner):
                return (
                    self._last_obs,
                    0.0,
                    True,
                    bool(truncated or done),
                    self._build_info({}),
                )

        mgr = self.inner._mgr
        assert mgr is not None
        prev_combat = mgr.get_combat_state()
        prev_hp = int(prev_combat.player.current_hp) if prev_combat is not None else 0
        run_action = to_runenv_combat_action(action)
        _obs, _run_reward, terminated, truncated, inner_info = self.inner.step(run_action)
        self._combat_steps += 1

        combat_now = mgr.get_combat_state() if mgr.phase == RunManager.PHASE_COMBAT else None
        scored = combat_now if combat_now is not None else prev_combat
        reward = 0.0
        if scored is not None:
            reward = float(compute_reward(scored, prev_hp))

        combat_ended = combat_now is None or (scored is not None and scored.is_over)
        if combat_ended and not (terminated or truncated):
            auto_done, auto_trunc = self._auto_noncombat()
            terminated = terminated or auto_done
            truncated = truncated or auto_trunc

        if is_combat_phase(self.inner):
            self._last_obs = combat_observation(self.inner)
            terminated = False
        elif scored is not None:
            try:
                self._last_obs = encode_observation(scored)
            except Exception:
                pass

        return (
            self._last_obs,
            reward,
            bool(terminated or _run_over(self.inner)),
            bool(truncated),
            self._build_info(inner_info),
        )

    def action_masks(self) -> np.ndarray:
        return combat_action_mask(self.inner)

    def close(self):
        self.inner.close()

    def render(self):
        return self.inner.render()

    def _auto_noncombat(self) -> tuple[bool, bool]:
        """Step hang-protocol noncombat until combat, run over, or cap.

        Returns ``(run_over, truncated)``.  These steps are not PPO transitions.
        """
        auto = 0
        truncated = False
        while not is_combat_phase(self.inner) and not _run_over(self.inner):
            if auto >= self.max_auto_steps or self.inner._step_count >= self.inner.max_steps:
                truncated = True
                break
            mask = self.inner.action_masks()
            try:
                action, _log = choose_jev_noncombat(
                    self.inner,
                    mask,
                    self._rng,
                    self.jev_adapter,
                    flags=self.jev_flags,
                )
            except Exception:
                valid = np.flatnonzero(np.asarray(mask) == 1)
                action = int(self._rng.choice(valid)) if valid.size else 0
            _obs, _reward, terminated, trunc, _info = self.inner.step(int(action))
            self._noncombat_auto_steps += 1
            auto += 1
            if terminated:
                return True, False
            if trunc:
                return False, True
        return _run_over(self.inner), truncated

    def _build_info(self, inner_info: dict[str, Any]) -> dict[str, Any]:
        mgr = getattr(self.inner, "_mgr", None)
        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "phase": mgr.phase if mgr is not None else None,
            "inner_phase": inner_info.get("phase"),
            "combat_steps": self._combat_steps,
            "noncombat_auto_steps": self._noncombat_auto_steps,
            "start_with_neow": HANG_START_WITH_NEOW,
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "jev_phases": HANG_JEV_PHASES,
            "combat_obs_size": OBS_SIZE,
            "loadout": None,
        }
        if inner_info:
            for key in ("act", "floor", "hp", "max_hp", "gold", "deck_size", "step"):
                if key in inner_info:
                    info[key] = inner_info[key]
        return info
