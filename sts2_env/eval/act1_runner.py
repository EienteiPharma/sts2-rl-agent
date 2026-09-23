"""Act1 RunEnv eval runner: model load, action choice, and episode loop.

Policy / Jev / map_lowhp behavior is unchanged; this module only owns the
eval wiring previously in ``scripts/eval_act1_runenv.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.eval.act1_suite import (
    JEV_SHADOW_SKIPPED,
    JEV_SHADOW_STUB,
    SEED_COUNT,
)
from sts2_env.eval.jev_client import build_jev_adapter
from sts2_env.eval.jev_config import (
    DEFAULT_JEV_FLAGS,
    JevPolicyFlags,
    resolve_jev_flags,
)
from sts2_env.eval.jev_policy import choose_jev_noncombat
from sts2_env.eval.jev_types import (
    EVENT_OPTIONS_EMPTY_REASON,
    EVENT_SAFE_FALLBACK_REASON,
    JEV_EVENT_OFF_REASON,
    POTION_OR_RELIC_REASON,
    POTION_OR_RELIC_SAFE_REASON,
    JevAnswer,
)
from sts2_env.eval.map_lowhp import (
    MAP_LOWHP_HARD_REASON,
    MAP_LOWHP_SOFT_B_REASON,
)
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.run_env import (
    RUN_OBS_SIZE,
    STS2RunEnv,
    _COMBAT_SIZE,
    _COMBAT_START,
)
from sts2_env.run.run_manager import RunManager


def model_obs_dim(model: Any) -> int:
    """Read a loaded SB3 model's observation width."""
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def require_obs_dim(model: Any, expected: int, what: str) -> int:
    """Refuse a zip whose observation width does not match *expected*."""
    obs_dim = model_obs_dim(model)
    if obs_dim != expected:
        raise SystemExit(
            f"{what} obs_dim={obs_dim} != expected={expected}; "
            "combat-suite zips are not valid for --policy model (use "
            "--policy hierarchical --combat-model), and RunEnv zips are not "
            "valid as --combat-model"
        )
    return obs_dim


def load_maskable_ppo(path: str) -> Any:
    zip_path = Path(path)
    if not zip_path.is_file():
        raise SystemExit(f"model zip not found: {path}")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib is required to load MaskablePPO zips. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    return MaskablePPO.load(str(zip_path), device="cpu")


def _run_manager(env: STS2RunEnv) -> RunManager:
    mgr = getattr(env, "_mgr", None)
    if mgr is None:
        raise RuntimeError("STS2RunEnv has no run manager; call reset() first")
    return mgr


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


def jev_shadow_fields(phase: str, *, jev_enabled: bool = False) -> dict[str, Any]:
    """Shadow log for steps that did not run a live Jev decision."""
    if phase == RunManager.PHASE_COMBAT:
        return JevAnswer(status=JEV_SHADOW_SKIPPED).as_log()
    if jev_enabled:
        return JevAnswer(status=JEV_SHADOW_STUB, fallback_reason="non_decision").as_log()
    return JevAnswer(status=JEV_SHADOW_STUB, fallback_reason="jev_off").as_log()


def _legal_random(mask: np.ndarray, rng: np.random.RandomState) -> int:
    valid = np.flatnonzero(np.asarray(mask) == 1)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def choose_hierarchical_action(
    env: STS2RunEnv,
    obs: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    combat_model: Any,
    *,
    received_obs_widths: list[int] | None = None,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
) -> tuple[int, dict[str, Any]]:
    """Pick a RunEnv action for hierarchical policy.

    Combat: encode *combat* obs (never ``obs`` / RunEnv) and predict, then map
    the combat action index into the RunEnv combat slice.
    Non-combat: legal random when ``jev_enabled`` is false; Jev Choice/Score
    when true (errors fall back to legal random). Ordinary EVENT is off unless
    ``jev_flags.allows_event()``. Detected Neow uses Jev only when
    ``jev_flags.allows_neow()`` is true (hang default: random boon).
    """
    mgr = _run_manager(env)
    phase = mgr.phase
    shadow = jev_shadow_fields(phase, jev_enabled=jev_enabled)

    if phase != RunManager.PHASE_COMBAT:
        if jev_enabled:
            adapter = jev_adapter or build_jev_adapter(enabled=True)
            return choose_jev_noncombat(
                env, mask, rng, adapter, flags=jev_flags or DEFAULT_JEV_FLAGS
            )
        return _legal_random(mask, rng), shadow

    combat = mgr.get_combat_state()
    if combat is None:
        return _legal_random(mask, rng), shadow

    combat_obs = encode_observation(combat)
    if combat_obs.shape[-1] != OBS_SIZE:
        raise RuntimeError(
            f"combat obs width {combat_obs.shape[-1]} != OBS_SIZE={OBS_SIZE}"
        )
    if np.asarray(obs).shape[-1] == combat_obs.shape[-1]:
        raise RuntimeError(
            "RunEnv obs width equals combat OBS_SIZE; hierarchical eval "
            "refuses to treat run observations as combat observations"
        )
    if received_obs_widths is not None:
        received_obs_widths.append(int(combat_obs.shape[-1]))

    owner = _selected_combat_owner(mgr, combat)
    combat_mask = get_action_mask(combat, owner=owner)
    local, _ = combat_model.predict(
        combat_obs, action_masks=combat_mask, deterministic=True
    )
    local = int(local)
    local = max(0, min(local, _COMBAT_SIZE - 1))
    return _COMBAT_START + local, shadow


def choose_action(
    policy: str,
    env: STS2RunEnv,
    obs: np.ndarray,
    info: dict[str, Any],
    mask: np.ndarray,
    rng: np.random.RandomState,
    model: Any,
    combat_model: Any,
    *,
    received_obs_widths: list[int] | None = None,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return ``(action, shadow_fields)`` for the current step."""
    if policy == "random":
        return _legal_random(mask, rng), jev_shadow_fields(info.get("phase", ""))
    if policy == "model":
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action), jev_shadow_fields(info.get("phase", ""))
    if policy == "hierarchical":
        return choose_hierarchical_action(
            env,
            obs,
            mask,
            rng,
            combat_model,
            received_obs_widths=received_obs_widths,
            jev_enabled=jev_enabled,
            jev_adapter=jev_adapter,
            jev_flags=jev_flags,
        )
    raise SystemExit(f"unknown policy: {policy}")


def _run_episode(
    env: STS2RunEnv,
    policy: str,
    model: Any,
    combat_model: Any,
    seed: int,
    rng: np.random.RandomState,
    *,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
    start_with_neow: bool = False,
) -> dict:
    obs, info = env.reset(seed=seed, options={"start_with_neow": start_with_neow})
    done = False
    ep_rew = 0.0
    steps = 0
    combat_steps = 0
    noncombat_steps = 0
    max_act = int(info.get("act", 0))
    reward = 0.0
    terminated = False
    truncated = False
    last_shadow = jev_shadow_fields(info.get("phase", ""), jev_enabled=jev_enabled)
    map_lowhp_hard_n = 0
    map_lowhp_soft_b_n = 0
    event_jev_used_n = 0
    event_safe_fallback_n = 0
    event_low_conf_random_n = 0
    event_options_empty_n = 0
    event_off_random_n = 0
    potion_or_relic_safe_n = 0
    potion_or_relic_random_n = 0
    while not done:
        mask = info.get("action_mask")
        if mask is None:
            mask = env.action_masks()
        mask = np.asarray(mask)
        action, last_shadow = choose_action(
            policy,
            env,
            obs,
            info,
            mask,
            rng,
            model,
            combat_model,
            jev_enabled=jev_enabled,
            jev_adapter=jev_adapter,
            jev_flags=jev_flags,
        )
        reason = last_shadow.get("shadow_fallback_reason")
        dec = last_shadow.get("shadow_decision")
        status = last_shadow.get("shadow_status")
        if (
            reason == MAP_LOWHP_HARD_REASON
            or last_shadow.get("map_lowhp_hard")
        ):
            map_lowhp_hard_n += 1
            print(
                f"map_lowhp_hard seed={seed} n={map_lowhp_hard_n} "
                f"executed={last_shadow.get('executed_id')} "
                f"jev_choice={last_shadow.get('shadow_suggestion')}",
                file=sys.stderr,
                flush=True,
            )
        if (
            reason == MAP_LOWHP_SOFT_B_REASON
            or last_shadow.get("map_lowhp_soft_b")
        ):
            map_lowhp_soft_b_n += 1
            print(
                f"map_lowhp_soft_b seed={seed} n={map_lowhp_soft_b_n} "
                f"executed={last_shadow.get('executed_id')} "
                f"jev_choice={last_shadow.get('shadow_suggestion')}",
                file=sys.stderr,
                flush=True,
            )
        if dec == "event_choice" or last_shadow.get("phase") == "EVENT":
            if status == "ok" and not reason:
                event_jev_used_n += 1
            elif reason == EVENT_SAFE_FALLBACK_REASON or last_shadow.get("event_safe_fallback"):
                event_safe_fallback_n += 1
            elif reason == "low_confidence_random":
                event_low_conf_random_n += 1
            elif reason == EVENT_OPTIONS_EMPTY_REASON:
                event_options_empty_n += 1
            elif reason == JEV_EVENT_OFF_REASON:
                event_off_random_n += 1
        if dec == "potion_or_relic_reward":
            if reason == POTION_OR_RELIC_SAFE_REASON or last_shadow.get("potion_or_relic_safe_fallback"):
                potion_or_relic_safe_n += 1
            elif reason == POTION_OR_RELIC_REASON:
                potion_or_relic_random_n += 1
        if info.get("phase") == RunManager.PHASE_COMBAT:
            combat_steps += 1
        else:
            noncombat_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += float(reward)
        steps += 1
        max_act = max(max_act, int(info.get("act", 0)))
        done = terminated or truncated
    return {
        "seed": seed,
        "act1_clear": bool(max_act >= 1),
        "full_run_win": bool(terminated and reward > 0),
        "truncated": bool(truncated),
        "max_act": max_act,
        "floor": int(info.get("floor", 0)),
        "hp": int(info.get("hp", 0)),
        "max_hp": int(info.get("max_hp", 0)),
        "gold": int(info.get("gold", 0)),
        "steps": steps,
        "combat_steps": combat_steps,
        "noncombat_steps": noncombat_steps,
        "reward": ep_rew,
        "shadow_suggestion": last_shadow.get("shadow_suggestion"),
        "shadow_status": last_shadow.get("shadow_status"),
        "shadow_confidence": last_shadow.get("shadow_confidence"),
        "shadow_hp_pressure": last_shadow.get("shadow_hp_pressure"),
        "shadow_fallback_reason": last_shadow.get("shadow_fallback_reason"),
        "map_lowhp_hard_n": map_lowhp_hard_n,
        "map_lowhp_soft_b_n": map_lowhp_soft_b_n,
        "event_jev_used_n": event_jev_used_n,
        "event_safe_fallback_n": event_safe_fallback_n,
        "event_low_conf_random_n": event_low_conf_random_n,
        "event_options_empty_n": event_options_empty_n,
        "event_off_random_n": event_off_random_n,
        "potion_or_relic_safe_n": potion_or_relic_safe_n,
        "potion_or_relic_random_n": potion_or_relic_random_n,
        "jev_card_fit": last_shadow.get("jev_card_fit"),
    }


def validate_policy_args(args: argparse.Namespace) -> None:
    if getattr(args, "strategic", None) == "jev":
        args.jev = "on"
    if args.policy == "model":
        if not args.model:
            raise SystemExit("--model required for --policy model")
        if args.combat_model:
            raise SystemExit("--combat-model is only valid with --policy hierarchical")
    elif args.policy == "hierarchical":
        if args.model and args.combat_model and args.model != args.combat_model:
            raise SystemExit(
                "hierarchical: --model and --combat-model are the same combat "
                "zip; pass one path (Surplus: --model)"
            )
        zip_path = args.combat_model or args.model
        if not zip_path:
            raise SystemExit(
                "--model (hung combat zip, obs_v1=181) required for "
                "--policy hierarchical; --combat-model is an alias"
            )
        args.combat_model = zip_path
    elif args.combat_model or args.model:
        raise SystemExit("--model/--combat-model require --policy model or hierarchical")
    if args.jev == "on" and args.policy != "hierarchical":
        raise SystemExit("--jev on is only valid with --policy hierarchical")
    if args.policy != "hierarchical":
        if args.jev_event == "on":
            raise SystemExit("--jev-event on is only valid with --policy hierarchical")
        if args.jev_neow == "on":
            raise SystemExit("--jev-neow on is only valid with --policy hierarchical")
    if int(getattr(args, "n", SEED_COUNT)) < 1:
        raise SystemExit("--n must be >= 1")
    args.jev_flags = resolve_jev_flags(
        jev_event=args.jev_event,
        jev_phases=args.jev_phases,
        jev_neow=args.jev_neow,
        map_lowhp=getattr(args, "map_lowhp", "on"),
        map_lowhp_hard=getattr(args, "map_lowhp_hard", "off"),
        map_lowhp_soft_b=getattr(args, "map_lowhp_soft_b", "off"),
    )


def load_policy_models(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load and type-check zips.

    ``--policy model`` requires RunEnv obs (201); combat zips are rejected.
    ``--policy hierarchical`` loads the hung combat zip (obs_v1=181) from
    ``--model`` or alias ``--combat-model``. Never feed RunEnv obs to it.
    """
    model = None
    combat_model = None
    if args.policy == "model":
        model = load_maskable_ppo(args.model)
        require_obs_dim(model, RUN_OBS_SIZE, "RunEnv --model")
    elif args.policy == "hierarchical":
        combat_path = args.combat_model or args.model
        combat_model = load_maskable_ppo(combat_path)
        require_obs_dim(combat_model, OBS_SIZE, "hierarchical combat zip (--model)")
    return model, combat_model


__all__ = [
    "_legal_random",
    "_run_episode",
    "_run_manager",
    "_selected_combat_owner",
    "choose_action",
    "choose_hierarchical_action",
    "jev_shadow_fields",
    "load_maskable_ppo",
    "load_policy_models",
    "model_obs_dim",
    "require_obs_dim",
    "validate_policy_args",
]
