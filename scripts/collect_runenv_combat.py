#!/usr/bin/env python3
"""Collect hang-protocol RunEnv combat transitions to a disk buffer.

**Default (unchanged):** non-combat auto-steps with hang Jev + TypeSafe.
**Opt-in colab_v1 / Pharma lock:** ``--jev off --noncombat-policy ppo
--combat-policy ppo --policy-zip …/bh_v1/final_model.zip`` — zero TypeSafe.

Usage:
    python scripts/collect_runenv_combat.py --dry-run
    python scripts/collect_runenv_combat.py \\
        --out output/runenv_combat_buffer/transitions.npz \\
        --n-envs 1 --n-steps 32 --policy random
    python scripts/collect_runenv_combat.py \\
        --jev off --noncombat-policy ppo --combat-policy ppo \\
        --policy-zip output/combat_ppo_obs_v1_bh_v1/final_model.zip \\
        --out output/runenv_combat_buffer_colab_v1/transitions.npz \\
        --n-envs 8 --n-steps 500000

Never writes into ``bh_v1`` or ``runenv_combat_buffer_ep/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_env.gym_env.combat_buffer import (
    COLAB_V1_COLLECT_OUT,
    HUNG_OUTDIR_NAME,
    PROTECTED_COLLECT_DIR_NAMES,
    hang_protocol_meta,
    refuse_frozen_path,
)
from sts2_env.gym_env.combat_collect import (
    collect_parallel,
    split_worker_steps,
)
from sts2_env.gym_env.planning_buffer import PLANNING_DEFAULT_OUT
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    hang_jev_flags,
)

HUNG_COMBAT_ZIP = f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
DEFAULT_OUT = "output/runenv_combat_buffer/transitions.npz"
PROTOCOL_ID = "hang_protocol_runenv_combat_collect LOCKED 2026-09-22"
COLAB_V1_PROTOCOL_ID = "colab_v1_collect Jev=off TypeSafe=off bh_v1_ppo"
PLANNING_ONLY_USAGE = (
    "python scripts/collect_runenv_combat.py --planning-only --jev off "
    "--noncombat-policy ppo --combat-policy ppo "
    "--policy-zip /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip "
    "--n-envs 8 --n-steps 500000"
)


def _is_protected_combat_npz(path: str | Path) -> bool:
    out = Path(path).expanduser()
    if out.name != "transitions.npz":
        return False
    return any(name in out.parts for name in PROTECTED_COLLECT_DIR_NAMES)


def _resolve_combat_policy(args: argparse.Namespace) -> str:
    if args.combat_policy is not None:
        cp = args.combat_policy
        return "model" if cp == "ppo" else cp
    return "model" if args.policy == "model" else args.policy


def normalize_collect_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.no_jev:
        args.jev = "off"
    args.jev_enabled = args.jev == "on"
    if not args.jev_enabled and args.noncombat_policy == "jev":
        raise SystemExit("--noncombat-policy jev requires --jev on")
    if args.noncombat_policy is None:
        args.noncombat_policy = "jev" if args.jev_enabled else "ppo"
    args.combat_policy_resolved = _resolve_combat_policy(args)
    if args.policy_zip:
        args.model = args.policy_zip
    if not args.jev_enabled:
        args.noncombat_model = args.noncombat_model or args.model
        if args.combat_policy is None and args.policy == "random":
            args.policy = "model"
        args.combat_policy_resolved = _resolve_combat_policy(args)
    if getattr(args, "planning_only", False):
        args.jev = "off"
        args.jev_enabled = False
        args.no_planning = False
        args.planning_out_resolved = args.planning_out or PLANNING_DEFAULT_OUT
        args.planning_jsonl_resolved = (
            args.planning_jsonl
            or str(Path(args.planning_out_resolved).with_suffix(".jsonl"))
        )
    elif args.no_planning:
        args.planning_out_resolved = None
        args.planning_jsonl_resolved = None
    elif args.jev_enabled and not args.planning_out:
        args.planning_out_resolved = None
        args.planning_jsonl_resolved = None
    else:
        args.planning_out_resolved = args.planning_out or PLANNING_DEFAULT_OUT
        args.planning_jsonl_resolved = (
            args.planning_jsonl
            or str(Path(args.planning_out_resolved).with_suffix(".jsonl"))
        )
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll hang-protocol RunEnv; write combat-only transitions to npz. "
            "Default keeps hang Jev on. Use --jev off for colab_v1 (no TypeSafe)."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output npz (default {DEFAULT_OUT}; colab_v1: {COLAB_V1_COLLECT_OUT})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Combat transitions to collect (default 256 smoke; Surplus e.g. 500000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Parallel collectors. Default hang Jev: 2-4 workers recommended. "
            "colab_v1 (no TypeSafe): e.g. 8 workers."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("random", "model"),
        default="random",
        help="Combat policy (legacy). model=MaskablePPO zip",
    )
    parser.add_argument(
        "--combat-policy",
        choices=("random", "model", "ppo"),
        default=None,
        help="Combat policy alias (ppo == model)",
    )
    parser.add_argument(
        "--model",
        "--continue-from",
        dest="model",
        default=HUNG_COMBAT_ZIP,
        help=f"Combat zip when using model/ppo (default {HUNG_COMBAT_ZIP})",
    )
    parser.add_argument(
        "--policy-zip",
        dest="policy_zip",
        default=None,
        help="Alias for --model (colab_v1 bh_v1 path)",
    )
    parser.add_argument(
        "--jev",
        choices=("on", "off"),
        default="on",
        help="Non-combat Jev/TypeSafe (default on for ep recipes)",
    )
    parser.add_argument(
        "--no-jev",
        action="store_true",
        help="Shorthand for --jev off",
    )
    parser.add_argument(
        "--noncombat-policy",
        choices=("jev", "ppo", "random"),
        default=None,
        help="Non-combat auto-step policy (default jev when --jev on, else ppo)",
    )
    parser.add_argument(
        "--noncombat-model",
        default=None,
        help=f"Zip for --noncombat-policy ppo (default combat --model / {HUNG_COMBAT_ZIP})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hang env + flags; no npz write, no torch",
    )
    parser.add_argument(
        "--planning-out",
        default=None,
        help=(
            "Non-combat planning npz (default when --jev off: "
            f"{PLANNING_DEFAULT_OUT})"
        ),
    )
    parser.add_argument(
        "--planning-jsonl",
        default=None,
        help="Optional per-worker jsonl audit (phase/action/tag); default beside planning npz",
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="Skip planning side-channel (combat npz only)",
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help=(
            "Jev off: roll env for planning only; never write combat npz "
            f"(full colab_v1 combat buffer protected). Default planning out: "
            f"{PLANNING_DEFAULT_OUT}"
        ),
    )
    args = parser.parse_args(argv)
    return normalize_collect_args(args)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    from sts2_env.core.constants import ACTION_SPACE_SIZE
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

    if getattr(args, "planning_only", False):
        out = None
    else:
        out = refuse_frozen_path(args.out, what="buffer")
    flags = hang_jev_flags()
    pool: dict[str, Any]
    if args.jev_enabled:
        from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs

        load_typesafe_api_keys()
        pool = typesafe_key_pool_summary()
        n_envs_note = warn_n_envs(args.n_envs)
    else:
        pool = {"typesafe_key_count": 0, "typesafe_key_pool": "off", "recommended_n_envs_max": 0}
        n_envs_note = None

    env = RunEnvOnPolicyCombatEnv(
        max_steps=min(int(args.max_steps), 80),
        seed_offset=0,
        jev_enabled=args.jev_enabled,
        noncombat_policy=args.noncombat_policy,
    )
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    proto = env.hang_protocol()
    env.close()
    quotas = split_worker_steps(int(args.n_steps), int(args.n_envs))
    protocol = PROTOCOL_ID if args.jev_enabled else COLAB_V1_PROTOCOL_ID
    return {
        "protocol": protocol,
        "dry_run": True,
        "out": str(out) if out is not None else None,
        "planning_only": getattr(args, "planning_only", False),
        "planning_only_usage": (
            PLANNING_ONLY_USAGE if getattr(args, "planning_only", False) else None
        ),
        "combat_colab_v1_protected": COLAB_V1_COLLECT_OUT,
        "n_steps": int(args.n_steps),
        "n_envs": int(args.n_envs),
        "n_envs_note": n_envs_note,
        "worker_quotas": quotas,
        "policy": args.combat_policy_resolved,
        "combat_policy": args.combat_policy_resolved,
        "noncombat_policy": args.noncombat_policy,
        "model": args.model,
        "noncombat_model": args.noncombat_model or args.model,
        "jev_enabled": args.jev_enabled,
        "typesafe": "on" if args.jev_enabled else "off",
        "receipt": (
            f"Jev=off TypeSafe=off policy_zip={args.model}"
            if not args.jev_enabled
            else None
        ),
        "hang": {
            "jev": HANG_JEV if args.jev_enabled else "off",
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "start_with_neow": HANG_START_WITH_NEOW,
            "allows_event": flags.allows_event(),
            "allows_neow": flags.allows_neow(),
        },
        "env_hang": proto,
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SPACE_SIZE,
        "obs_shape": list(obs.shape),
        "mask_size": int(mask.shape[-1]),
        "phase": info.get("phase"),
        "jev": info.get("jev"),
        "typesafe_env": info.get("typesafe"),
        "start_with_neow": info.get("start_with_neow"),
        "loadout": info.get("loadout"),
        "keys": ["obs", "next_obs", "action", "reward", "done", "action_mask"],
        "typesafe_key_count": pool["typesafe_key_count"],
        "typesafe_key_pool": pool.get("typesafe_key_pool"),
        "recommended_n_envs_max": pool.get("recommended_n_envs_max"),
        "planning_out": args.planning_out_resolved,
        "planning_jsonl": args.planning_jsonl_resolved,
        "planning_schema": (
            "obs/next_obs (151) run; action/mask (157); phase_code; policy_tag"
            if args.planning_out_resolved
            else None
        ),
        "note": (
            "Jev stays on collect; train_combat_from_buffer.py is the learn half"
            if args.jev_enabled
            else "colab_v1: non-combat PPO fail-open random; zero TypeSafe calls"
        ),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    planning_only = bool(getattr(args, "planning_only", False))
    if planning_only:
        out = None
    elif _is_protected_combat_npz(args.out):
        raise SystemExit(
            f"refusing to overwrite protected full combat buffer {args.out}; "
            "use --planning-only to emit planning npz only"
        )
    else:
        out = refuse_frozen_path(args.out, what="buffer")
    combat_policy = args.combat_policy_resolved
    if combat_policy in ("model", "ppo") and not Path(args.model).is_file():
        raise SystemExit(f"collect combat zip not found: {args.model}")
    flags = hang_jev_flags()
    if args.jev_enabled:
        from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs

        pool = typesafe_key_pool_summary(load_typesafe_api_keys())
        note = warn_n_envs(args.n_envs)
    else:
        pool = {"typesafe_key_count": 0}
        note = None

    print(
        "Collecting planning-only (no combat npz write)"
        if planning_only
        else "Collecting hang-protocol RunEnv combat segments"
    )
    if not args.jev_enabled:
        print("Jev=off TypeSafe=off")
        print(f"  policy_zip: {args.model}")
        print("  assist_v3: off (collect path; no jev-turn)")
    if planning_only:
        print("  combat:    (protected; full colab_v1 buffer not touched)")
    else:
        print("  out:       ", out)
    print("  n_steps:   ", args.n_steps)
    print("  n_envs:    ", args.n_envs)
    if note:
        print("  n_envs:    ", note)
    print("  combat:    ", combat_policy, args.model)
    print("  noncombat: ", args.noncombat_policy, args.noncombat_model or args.model)
    if args.planning_out_resolved:
        print("  planning:  ", args.planning_out_resolved)
    if args.jev_enabled:
        print("  hang:      jev on / event off / neow off / start_with_neow")
        print("  allows_event:", flags.allows_event(), "allows_neow", flags.allows_neow())
        print(
            "  typesafe keys:",
            pool["typesafe_key_count"],
            "(box-secrets auto-load; values not printed)",
        )
    print()
    result = collect_parallel(
        out_path=out or COLAB_V1_COLLECT_OUT,
        n_steps=int(args.n_steps),
        n_envs=int(args.n_envs),
        policy=combat_policy,
        model=args.model if combat_policy in ("model", "ppo") else None,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        jev_enabled=args.jev_enabled,
        noncombat_policy=args.noncombat_policy,
        noncombat_model=args.noncombat_model or args.model,
        combat_policy=combat_policy,
        combat_model=args.model if combat_policy in ("model", "ppo") else None,
        planning_out=args.planning_out_resolved,
        planning_jsonl=args.planning_jsonl_resolved,
        planning_only=planning_only,
    )
    result["protocol"] = PROTOCOL_ID if args.jev_enabled else COLAB_V1_PROTOCOL_ID
    result["hang"] = hang_protocol_meta() if args.jev_enabled else {
        **hang_protocol_meta(),
        "jev": "off",
        "typesafe": "off",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args), ensure_ascii=False, indent=2))
        return
    collect(args)


if __name__ == "__main__":
    main()
