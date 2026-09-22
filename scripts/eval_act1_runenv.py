#!/usr/bin/env python3
"""Frozen Act1 RunEnv eval (box ops). Protocol: docs/act1_runenv_eval_protocol.md

Seeds 200000..200049 (50). Primary metric: act1_clear_rate (max act >= 1).
Never conflate with combat suite win rates.

Policies
--------
* ``random``: legal random among ``action_masks()==1``.
* ``model``: RunEnv-sized MaskablePPO only (obs == RUN_OBS_SIZE). Combat zips
  are rejected.
* ``hierarchical``: combat steps use ``--combat-model`` (combat OBS_SIZE /
  obs_v1) via ``encode_observation(CombatState)`` + combat ``get_action_mask``.
  Non-combat: ``--jev off`` (default) is legal random; ``--jev on`` calls
  TypeSafe/Jev Choice (and Score for rest_or_continue). Jev never runs in
  combat. Failures log ``error`` and fall back to legal random.

Never feed RunEnv observations into the combat model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.eval.jev import JevAnswer, build_jev_adapter
from sts2_env.eval.jev_policy import choose_jev_noncombat
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.run_env import (
    RUN_OBS_SIZE,
    STS2RunEnv,
    _COMBAT_SIZE,
    _COMBAT_START,
)
from sts2_env.run.run_manager import RunManager

SEED_START = 200000
SEED_COUNT = 50
SEEDS = list(range(SEED_START, SEED_START + SEED_COUNT))
PROTOCOL_ID = "act1_runenv_eval_protocol.md LOCKED 2026-09-22"

JEV_SHADOW_SKIPPED = "skipped"
JEV_SHADOW_STUB = "stub"


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    clears = [r for r in rows if r["act1_clear"]]
    floors = [r["floor"] for r in rows]
    return {
        "n": n,
        "act1_clear_rate": round(sum(r["act1_clear"] for r in rows) / n, 4) if n else 0.0,
        "full_run_win_rate": round(sum(r["full_run_win"] for r in rows) / n, 4) if n else 0.0,
        "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 4) if n else 0.0,
        "median_floor": float(np.median(floors)) if floors else 0.0,
        "mean_floor": round(float(np.mean(floors)), 3) if floors else 0.0,
        "mean_hp_on_clear": round(float(np.mean([r["hp"] for r in clears])), 2) if clears else None,
        "max_floor": int(max(floors)) if floors else 0,
    }


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
) -> tuple[int, dict[str, Any]]:
    """Pick a RunEnv action for hierarchical policy.

    Combat: encode *combat* obs (never ``obs`` / RunEnv) and predict, then map
    the combat action index into the RunEnv combat slice.
    Non-combat: legal random when ``jev_enabled`` is false; Jev Choice/Score
    when true (errors fall back to legal random).
    """
    mgr = _run_manager(env)
    phase = mgr.phase
    shadow = jev_shadow_fields(phase, jev_enabled=jev_enabled)

    if phase != RunManager.PHASE_COMBAT:
        if jev_enabled:
            adapter = jev_adapter or build_jev_adapter(enabled=True)
            return choose_jev_noncombat(env, mask, rng, adapter)
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
) -> dict:
    obs, info = env.reset(seed=seed)
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
        )
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
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Frozen Act1 RunEnv eval (seeds 200000..200049)")
    ap.add_argument(
        "--policy",
        choices=["random", "model", "hierarchical"],
        default="random",
        help="random | RunEnv MaskablePPO | combat zip in combat + Jev/random outside",
    )
    ap.add_argument(
        "--model",
        default="",
        help="MaskablePPO zip with RUN_OBS_SIZE input (not a combat zip)",
    )
    ap.add_argument(
        "--combat-model",
        default="",
        help="Combat MaskablePPO zip with OBS_SIZE/obs_v1 input (hierarchical only)",
    )
    ap.add_argument(
        "--jev",
        choices=["on", "off"],
        default="off",
        help="hierarchical non-combat: off=legal random (default); on=TypeSafe/Jev Choice",
    )
    ap.add_argument(
        "--strategic",
        choices=["jev"],
        default=None,
        help="Alias for --jev on when set to 'jev'",
    )
    ap.add_argument(
        "--out",
        default="/workspace/sts2-sim/evals/act1_runenv_latest.json",
    )
    ap.add_argument("--max-steps", type=int, default=2000)
    return ap.parse_args(argv)


def validate_policy_args(args: argparse.Namespace) -> None:
    if getattr(args, "strategic", None) == "jev":
        args.jev = "on"
    if args.policy == "model":
        if not args.model:
            raise SystemExit("--model required for --policy model")
        if args.combat_model:
            raise SystemExit("--combat-model is only valid with --policy hierarchical")
    elif args.policy == "hierarchical":
        if not args.combat_model:
            raise SystemExit("--combat-model required for --policy hierarchical")
        if args.model:
            raise SystemExit(
                "--model is the RunEnv path; hierarchical combat zips go on --combat-model"
            )
    elif args.combat_model or args.model:
        raise SystemExit("--model/--combat-model require --policy model or hierarchical")
    if args.jev == "on" and args.policy != "hierarchical":
        raise SystemExit("--jev on is only valid with --policy hierarchical")


def load_policy_models(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load and type-check zips. Combat zips must not be used as --model."""
    model = None
    combat_model = None
    if args.policy == "model":
        model = load_maskable_ppo(args.model)
        require_obs_dim(model, RUN_OBS_SIZE, "RunEnv --model")
    elif args.policy == "hierarchical":
        combat_model = load_maskable_ppo(args.combat_model)
        require_obs_dim(combat_model, OBS_SIZE, "hierarchical --combat-model")
    return model, combat_model


def build_report(
    *,
    policy: str,
    model_path: str,
    combat_model_path: str,
    rows: list[dict],
    elapsed_s: float,
    jev: str = "off",
) -> dict:
    summary = _summarize(rows)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_ID,
        "suite": "act1_runenv",
        "policy": policy,
        "model": model_path or None,
        "combat_model": combat_model_path or None,
        "jev": jev,
        "character": "Ironclad",
        "ascension": 0,
        "seeds": {
            "start": SEED_START,
            "count": SEED_COUNT,
            "list_head": SEEDS[:3],
            "list_tail": SEEDS[-3:],
        },
        "run_obs_size": RUN_OBS_SIZE,
        "combat_obs_size": OBS_SIZE,
        "jev_shadow": {
            "mode": "on" if jev == "on" else "stub",
            "note": (
                "Combat never uses Jev. --jev off: legal random + stub logs. "
                "--jev on: Choice/Score with confidence>=0.65 and hp_pressure "
                "rest/continue thresholds; errors fall back to legal random. "
                "Not an Act1-clear gate."
            ),
        },
        "elapsed_s": round(elapsed_s, 1),
        "summary": summary,
        "rows": rows,
    }


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    slim = {k: v for k, v in report.items() if k != "rows"}
    summary_path = out.with_name(out.stem + ".summary.json") if out.suffix == ".json" else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n")
    return summary_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_policy_args(args)
    model, combat_model = load_policy_models(args)
    jev_enabled = args.policy == "hierarchical" and args.jev == "on"
    jev_adapter = build_jev_adapter(enabled=jev_enabled) if args.policy == "hierarchical" else None

    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=args.max_steps)
    rng = np.random.RandomState(0)
    rows: list[dict] = []
    t0 = datetime.now(timezone.utc)
    for seed in SEEDS:
        rows.append(
            _run_episode(
                env,
                args.policy,
                model,
                combat_model,
                seed,
                rng,
                jev_enabled=jev_enabled,
                jev_adapter=jev_adapter,
            )
        )
    env.close()
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    report = build_report(
        policy=args.policy,
        model_path=args.model,
        combat_model_path=args.combat_model,
        rows=rows,
        elapsed_s=elapsed,
        jev=args.jev if args.policy == "hierarchical" else "off",
    )
    out = Path(args.out)
    write_report(report, out)
    slim = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
