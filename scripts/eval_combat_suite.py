#!/usr/bin/env python3
"""Combat suite eval. Hang HOLD: ``--suite loadout_v1 --n-eps 20``.

Protocol: ``docs/HOLD_PROTOCOL.md``. Dual gate overall ≥70 / Boss ≥40 on this
aligned protocol. Hang zip stays ``bh_v1``. Not Act1 RunEnv.
``--combat-policy jev`` is **experimental** (HOLD failed twice; latest
``ecd0073`` B 7.8/15.6/Boss 0.0; not hang / not next mainline); default
**ppo**. See ``docs/COMBAT_JEV_HOLD_FAIL.md``.

Box ops used to keep a bare Act1 22-enc copy at ``/workspace/sts2-sim/eval_combat_suite.py``
(no ``--suite loadout_v1``). That is **not** the hang table. Use this script.

Sentry 8-way smoke: ``--workers 8`` (default 1 = serial, same as before).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.eval.combat_hold import (
    HOLD_BOSS_MIN,
    HOLD_OVERALL_MIN,
    HUNG_COMBAT_ZIP,
    hold_protocol_meta,
    run_hold_smoke,
)

PROTOCOL_ID = "loadout_v1 HOLD LOCKED 2026-09-23"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combat suite eval (hang HOLD = --suite loadout_v1)"
    )
    parser.add_argument(
        "--suite",
        default="loadout_v1",
        help="Only loadout_v1 is the hang HOLD protocol (fixtures 01-03, enc 16-21)",
    )
    parser.add_argument(
        "--n-eps",
        type=int,
        default=20,
        help="Episodes per fixture×encounter (hang table used 20 → 360 fights)",
    )
    parser.add_argument(
        "--model",
        default=HUNG_COMBAT_ZIP,
        help="Combat MaskablePPO zip (obs_v1=181). Hang zip is bh_v1.",
    )
    parser.add_argument("--out", type=str, default="", help="Write summary JSON")
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 device: cpu (default, hang-shaped) | auto | cuda | cuda:N",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Gym steps per fight (hold smoke default)",
    )
    parser.add_argument(
        "--combat-policy",
        choices=["ppo", "jev", "jev-turn"],
        default="ppo",
        help=(
            "Combat source. Default ppo = hung bh_v1 zip (hang table). "
            "jev is experimental (failed stepwise HOLD bypass). "
            "jev-turn = combat_turn_plan Choice on plan_id (code-enumerated "
            "steps, replan caps); fail-open catastrophe only to bh_v1."
        ),
    )
    parser.add_argument(
        "--turn-plan-choice-cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "jev-turn: max plan_id options in one Jev Choice (default 32, "
            "override via STS2_TURN_PLAN_CHOICE_CAP; platform max 255)."
        ),
    )
    parser.add_argument(
        "--bh-assist",
        choices=["off", "on"],
        default="off",
        help=(
            "jev-turn only: inject bh_assist ranked_semantic + risk_notes into "
            "plan Choice state/prompt when checkpoint loads (fail-open to "
            "assist-off on missing ckpt or errors). Hang execution stays bh_v1."
        ),
    )
    parser.add_argument(
        "--bh-assist-ckpt",
        default=None,
        metavar="PATH",
        help=(
            "Assist ranker npz (default "
            "/workspace/sts2-sim/output/combat_bh_assist_v1/bh_assist_ranker.npz "
            "or STS2_BH_ASSIST_CKPT)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel HOLD fights (ProcessPool spawn, same as collect). "
            "Default 1 = serial (prior behavior). Sentry smoke passes --workers 8. "
            "experimental jev arm shards TypeSafe keys "
            "(TYPESAFE_API_KEY / TYPESAFE_API_KEY_1.._4 / TYPESAFE_API_KEYS)."
        ),
    )
    return parser.parse_args(argv)


def resolve_device(spec: str, *, cuda_available: bool | None = None) -> str:
    s = (spec or "cpu").strip().lower()
    if s == "cpu":
        return "cpu"
    if s == "auto":
        if cuda_available is None:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
        return "cuda" if cuda_available else "cpu"
    if s == "cuda" or s.startswith("cuda:"):
        if cuda_available is None:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
        if not cuda_available:
            raise SystemExit(f"--device {spec} requested but CUDA unavailable")
        return s
    raise SystemExit(f"unknown --device {spec!r} (use auto|cuda|cpu|cuda:N)")


def load_maskable_ppo(path: str, *, device: str) -> Any:
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
    return MaskablePPO.load(str(zip_path), device=device)


def model_obs_dim(model: Any) -> int:
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def pct(rate: float) -> str:
    return f"{100.0 * float(rate):.1f}"


def empty_combat_jev_report(*, n_episodes: int = 0) -> dict[str, Any]:
    """PPO-path combat_jev payload without importing combat_jev (no import cycle)."""
    del n_episodes  # mean is 0 with no calls
    return {
        "jev_calls": 0,
        "jev_failopen": 0,
        "failopen_rate": 0.0,
        "latency_ms": {"p50": None, "p95": None, "n": 0, "mean": None},
        "combat_jev_calls_mean": 0.0,
        "jev_failopen_reason": {
            "timeout": 0,
            "error": 0,
            "bad_id": 0,
            "low_conf": 0,
            "empty_list": 0,
        },
        "turn_plan_turns": 0,
        "turn_plan_fulfilled": 0,
        "turn_plan_catastrophe_failopen": 0,
        "jev_fulfilled_rate": 0.0,
        "catastrophe_failopen_rate": 0.0,
        "jev_turn_catastrophe_reason": {
            "timeout": 0,
            "error": 0,
            "empty_list": 0,
            "illegal_plan": 0,
            "replan_cap": 0,
        },
        "note": (
            "Combat-Jev is an optional bypass (--combat-policy jev), not a "
            "hang swap. Default remains ppo/bh_v1. Conf min 0.35."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = str(args.suite).strip()
    if suite != "loadout_v1":
        raise SystemExit(
            f"unknown --suite {suite!r}. Hang HOLD is --suite loadout_v1 "
            "(fixtures 01-03, enc 16-21, relics/potions applied). "
            "Bare Act1 22-enc is a different protocol and is not this table."
        )
    n_eps = int(args.n_eps)
    if n_eps < 1:
        raise SystemExit("--n-eps must be >= 1")
    workers = int(getattr(args, "workers", 1) or 1)
    if workers < 1:
        raise SystemExit("--workers must be >= 1")
    device = resolve_device(args.device)
    # Cards must register before observation imports CombatState.
    import sts2_env.cards  # noqa: F401
    from sts2_env.gym_env.observation import OBS_SIZE

    model = load_maskable_ppo(args.model, device=device)
    obs_dim = model_obs_dim(model)
    if obs_dim != OBS_SIZE:
        raise SystemExit(
            f"combat zip obs_dim={obs_dim} != OBS_SIZE={OBS_SIZE} (obs_v1)"
        )

    def predict_fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    combat_policy = str(getattr(args, "combat_policy", "ppo") or "ppo")
    from sts2_env.eval.combat_turn_plan import resolve_turn_plan_choice_cap

    turn_plan_choice_cap = resolve_turn_plan_choice_cap(
        getattr(args, "turn_plan_choice_cap", None)
    )
    from sts2_env.eval.bh_assist import resolve_turn_plan_bh_assist_from_flags

    bh_assist_mode = str(getattr(args, "bh_assist", "off") or "off")
    bh_assist_config = resolve_turn_plan_bh_assist_from_flags(
        bh_assist=bh_assist_mode,
        bh_assist_ckpt=getattr(args, "bh_assist_ckpt", None),
    )
    choose_fn = None
    combat_jev_summary = None
    if workers == 1 and combat_policy == "jev":
        from sts2_env.eval.combat_jev import CombatJevTelemetry, choose_combat_step
        from sts2_env.eval.jev_client import build_jev_adapter

        adapter = build_jev_adapter(enabled=True)
        telemetry = CombatJevTelemetry()
        rng = np.random.RandomState(0)

        def choose_fn(env, obs, mask):
            combat = getattr(env, "combat", None)
            if combat is None:
                return predict_fn(obs, mask)
            local, _shadow = choose_combat_step(
                combat,
                mask,
                rng,
                model,
                adapter=adapter,
                combat_obs=obs,
                telemetry=telemetry,
            )
            return int(local)

        combat_jev_summary = telemetry
    elif workers == 1 and combat_policy == "jev-turn":
        from sts2_env.eval.combat_jev import CombatJevTelemetry
        from sts2_env.eval.combat_turn_plan import choose_combat_turn_plan_action
        from sts2_env.eval.jev_client import build_jev_adapter

        adapter = build_jev_adapter(enabled=True)
        rng = np.random.RandomState(0)
        telemetry = CombatJevTelemetry()

        def choose_fn(env, obs, mask):
            combat = getattr(env, "combat", None)
            if combat is None:
                return predict_fn(obs, mask)
            local, _shadow = choose_combat_turn_plan_action(
                combat,
                mask,
                rng,
                model,
                adapter=adapter,
                combat_obs=obs,
                env=env,
                telemetry=telemetry,
                turn_plan_choice_cap=turn_plan_choice_cap,
                bh_assist_config=bh_assist_config,
            )
            return int(local)

        combat_jev_summary = telemetry

    summary = run_hold_smoke(
        predict_fn,
        n_eps=n_eps,
        max_steps=int(args.max_steps),
        choose_fn=choose_fn,
        workers=workers,
        model_path=str(args.model),
        device=device,
        combat_policy=combat_policy,
        turn_plan_choice_cap=turn_plan_choice_cap,
        bh_assist=bh_assist_mode if combat_policy == "jev-turn" else None,
        bh_assist_ckpt=(
            bh_assist_config.ckpt_path
            if combat_policy == "jev-turn" and bh_assist_config is not None
            else None
        ),
    )
    n_fights = int(summary["overall"]["n"])
    if combat_jev_summary is not None:
        combat_jev_payload = combat_jev_summary.as_report(n_episodes=n_fights)
    elif combat_policy in ("jev", "jev-turn") and summary.get("combat_jev_telemetry"):
        from sts2_env.eval.combat_jev import CombatJevTelemetry

        combat_jev_payload = CombatJevTelemetry.from_dict(
            summary["combat_jev_telemetry"]
        ).as_report(n_episodes=n_fights)
    else:
        combat_jev_payload = empty_combat_jev_report(n_episodes=n_fights)

    overall = float(summary["overall"]["win_rate"])
    elite = float(summary["elite"]["win_rate"])
    boss = float(summary["boss"]["win_rate"])
    payload = {
        "protocol": PROTOCOL_ID,
        "suite": "loadout_v1",
        "model": str(Path(args.model).resolve()),
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **hold_protocol_meta(n_eps=n_eps),
        "overall": summary["overall"],
        "elite": summary["elite"],
        "boss": summary["boss"],
        "gate": summary["gate"],
        "passed": bool(summary["passed"]),
        "n_eps": n_eps,
        "workers": workers,
        "combat_policy": combat_policy,
        "turn_plan_choice_cap": (
            turn_plan_choice_cap if combat_policy == "jev-turn" else None
        ),
        "bh_assist": bh_assist_mode if combat_policy == "jev-turn" else None,
        "bh_assist_ckpt": (
            bh_assist_config.ckpt_path
            if combat_policy == "jev-turn"
            and bh_assist_config is not None
            and bh_assist_config.enabled
            else None
        ),
        "combat_jev": combat_jev_payload,
    }
    text = (
        f"HOLD loadout_v1 n_eps={n_eps} workers={workers} "
        f"combat_policy={combat_policy}  "
        f"{pct(overall)} / {pct(elite)} / Boss {pct(boss)}  "
        f"passed={payload['passed']} "
        f"(gate {HOLD_OVERALL_MIN:.0%}/{HOLD_BOSS_MIN:.0%})"
    )
    print(text)
    hang = payload["hang_table"]
    print(
        "  hang 2026-09-22 bh_v1 table: "
        f"{pct(hang['overall'])} / {pct(hang['elite'])} / Boss {pct(hang['boss'])} "
        "(n_eps=20; tolerance ~±3–5pp)"
    )
    out = str(args.out).strip()
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
