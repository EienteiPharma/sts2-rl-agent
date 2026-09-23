#!/usr/bin/env python3
"""Collect hang-protocol RunEnv combat transitions to a disk buffer.

Non-combat is auto-stepped with locked hang Jev (MAP/REST/CARD live,
EVENT off, Neow random, --start-with-neow). Only combat (obs, action,
reward, done, action_mask) is written. TypeSafe stays on this collect
path; ``train_combat_from_buffer.py`` learns without Jev.

Usage:
    python scripts/collect_runenv_combat.py --dry-run
    python scripts/collect_runenv_combat.py \\
        --out output/runenv_combat_buffer/transitions.npz \\
        --n-envs 1 --n-steps 32 --policy random
    python scripts/collect_runenv_combat.py \\
        --out output/runenv_combat_buffer/transitions.npz \\
        --n-envs 4 --n-steps 50000 --policy model \\
        --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip

Never writes into ``bh_v1``. Hang flags/bars unchanged.
TypeSafe pool loads from box-secrets card.TYPESAFE_API_KEY + _1..4
automatically (no user re-paste; process env need not be pre-injected).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_env.gym_env.combat_buffer import (
    HUNG_OUTDIR_NAME,
    collect_parallel,
    hang_protocol_meta,
    refuse_frozen_path,
    split_worker_steps,
)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll hang-protocol RunEnv; write combat-only transitions to npz. "
            "Does not turn Jev off. Does not overwrite bh_v1."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output npz (default {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Combat transitions to collect (default 256 smoke; Surplus e.g. 50000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Parallel hang-protocol collectors (each worker still Jev). "
            "First recipe 2-4, not 16. Optional TypeSafe key pool: one key per env."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("random", "model"),
        default="random",
        help="Combat action policy. random=legal mask (no torch). model=MaskablePPO zip",
    )
    parser.add_argument(
        "--model",
        "--continue-from",
        dest="model",
        default=HUNG_COMBAT_ZIP,
        help=f"Combat zip when --policy model (default {HUNG_COMBAT_ZIP})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hang env + flags; no npz write, no torch",
    )
    return parser.parse_args(argv)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    from sts2_env.core.constants import ACTION_SPACE_SIZE
    from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

    out = refuse_frozen_path(args.out, what="buffer")
    flags = hang_jev_flags()
    load_typesafe_api_keys()
    env = RunEnvOnPolicyCombatEnv(max_steps=min(int(args.max_steps), 80), seed_offset=0)
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    proto = env.hang_protocol()
    env.close()
    quotas = split_worker_steps(int(args.n_steps), int(args.n_envs))
    pool = typesafe_key_pool_summary()
    return {
        "protocol": PROTOCOL_ID,
        "dry_run": True,
        "out": str(out),
        "n_steps": int(args.n_steps),
        "n_envs": int(args.n_envs),
        "n_envs_note": warn_n_envs(args.n_envs),
        "worker_quotas": quotas,
        "policy": args.policy,
        "model": args.model,
        "hang": {
            "jev": HANG_JEV,
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
        "jev_event": info.get("jev_event"),
        "jev_neow": info.get("jev_neow"),
        "start_with_neow": info.get("start_with_neow"),
        "loadout": info.get("loadout"),
        "keys": ["obs", "next_obs", "action", "reward", "done", "action_mask"],
        "typesafe_key_count": pool["typesafe_key_count"],
        "typesafe_key_pool": pool["typesafe_key_pool"],
        "recommended_n_envs_max": pool["recommended_n_envs_max"],
        "note": "Jev stays on collect; train_combat_from_buffer.py is the learn half",
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    out = refuse_frozen_path(args.out, what="buffer")
    if args.policy == "model" and not Path(args.model).is_file():
        raise SystemExit(f"collect --policy model zip not found: {args.model}")
    flags = hang_jev_flags()
    from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs

    pool = typesafe_key_pool_summary(load_typesafe_api_keys())
    note = warn_n_envs(args.n_envs)
    print("Collecting hang-protocol RunEnv combat segments")
    print("  out:       ", out)
    print("  n_steps:   ", args.n_steps)
    print("  n_envs:    ", args.n_envs)
    if note:
        print("  n_envs:    ", note)
    print("  policy:    ", args.policy)
    print("  hang:      jev on / event off / neow off / start_with_neow")
    print("  allows_event:", flags.allows_event(), "allows_neow", flags.allows_neow())
    print("  typesafe keys:", pool["typesafe_key_count"], "(box-secrets auto-load; values not printed)")
    print()
    result = collect_parallel(
        out_path=out,
        n_steps=int(args.n_steps),
        n_envs=int(args.n_envs),
        policy=args.policy,
        model=args.model if args.policy == "model" else None,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
    )
    result["protocol"] = PROTOCOL_ID
    result["hang"] = hang_protocol_meta()
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
