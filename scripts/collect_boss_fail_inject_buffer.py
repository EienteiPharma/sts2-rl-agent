#!/usr/bin/env python3
"""Collect combat buffer from Boss forward-inject jobs (hung PPO; no TypeSafe)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sts2_env.eval.boss_fail_inject_collect import (
    BOSS_FAIL_INJECT_BUFFER_PROTOCOL,
    DEFAULT_BOSS_FAIL_INJECT_BUFFER,
    collect_transitions_for_hold_jobs,
    dry_run_collect_manifest,
    resolve_inject_jobs,
)
from sts2_env.eval.combat_hold import HUNG_COMBAT_ZIP
from sts2_env.gym_env.combat_buffer import save_combat_buffer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect hang PPO buffer from boss_fail forward-inject jobs"
    )
    p.add_argument(
        "--pack-dir",
        default="evals/boss_fail_pack_v1",
        help="Boss fail pack (used if --inject-jobs omitted)",
    )
    p.add_argument(
        "--inject-jobs",
        default="",
        help="inject_jobs.json (default: <pack-dir>/inject_jobs.json if present)",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_BOSS_FAIL_INJECT_BUFFER,
        help=f"Output transitions.npz (default {DEFAULT_BOSS_FAIL_INJECT_BUFFER})",
    )
    p.add_argument("--model", default=HUNG_COMBAT_ZIP, help="Hung bh_v1 MaskablePPO zip")
    p.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Max jobs to roll (0 = all)",
    )
    p.add_argument("--max-steps", type=int, default=400, help="Gym steps per fight")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def load_ppo_predict(model_path: str, device: str):
    zip_path = Path(model_path)
    if not zip_path.is_file():
        raise SystemExit(f"model zip not found: {model_path}")
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(zip_path), device=device)

    def predict_fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return predict_fn


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inject_path = str(args.inject_jobs).strip()
    pack_dir = str(args.pack_dir).strip()
    if not inject_path:
        candidate = Path(pack_dir) / "inject_jobs.json"
        if candidate.is_file():
            inject_path = str(candidate)
    jobs, inject_meta = resolve_inject_jobs(
        inject_jobs_path=inject_path or None,
        pack_dir=pack_dir if not inject_path else None,
    )
    if int(args.max_jobs) > 0:
        jobs = jobs[: int(args.max_jobs)]

    if args.dry_run:
        manifest = dry_run_collect_manifest(
            inject_jobs_path=inject_path or pack_dir,
            n_jobs=len(jobs),
            max_steps=int(args.max_steps),
            model_path=str(args.model),
            pack_dir=pack_dir,
            out_path=args.out,
        )
        manifest["inject_meta"] = inject_meta
        print(json.dumps(manifest, indent=2))
        return 0

    import sts2_env.cards  # noqa: F401

    predict_fn = load_ppo_predict(str(args.model), str(args.device))
    arrays = collect_transitions_for_hold_jobs(
        jobs, predict_fn, max_steps=int(args.max_steps)
    )
    meta = {
        "protocol": BOSS_FAIL_INJECT_BUFFER_PROTOCOL,
        "inject_jobs": inject_path or None,
        "pack_dir": pack_dir,
        "n_jobs": len(jobs),
        "max_steps": int(args.max_steps),
        "model": str(Path(args.model).resolve()),
        "policy": "ppo_hung_bh_v1",
        "inject_meta_protocol": inject_meta.get("protocol"),
    }
    out = save_combat_buffer(args.out, arrays, meta)
    print(f"buffer: {out} n={arrays['obs'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
