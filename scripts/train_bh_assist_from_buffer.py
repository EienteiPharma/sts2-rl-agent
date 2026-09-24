#!/usr/bin/env python3
"""Train combat ``bh_assist`` advisory ranker from hang combat buffer (not hang PPO).

Usage:
    python scripts/train_bh_assist_from_buffer.py --dry-run
    python scripts/train_bh_assist_from_buffer.py \\
        --buffer output/runenv_combat_buffer/transitions.npz \\
        --output-dir output/combat_bh_assist_v1 \\
        --train-steps 128

Never writes ``combat_ppo_obs_v1_bh_v1`` / frozen hang outdirs. No MaskablePPO learn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_env.eval.bh_assist_train import (
    DEFAULT_BH_ASSIST_OUTDIR,
    PROTOCOL_ID,
    build_assist_rows_from_buffer,
    dry_run_manifest,
    load_assist_rows_from_buffer_path,
    refuse_bh_assist_output_path,
    refuse_hang_ppo_continue,
    save_assist_checkpoint,
    train_linear_assist_ranker,
)
from sts2_env.gym_env.combat_buffer import synthetic_combat_buffer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bh_assist ranker from combat buffer")
    p.add_argument(
        "--buffer",
        default="",
        help="Hang combat buffer npz (optional; synthetic if omitted)",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_BH_ASSIST_OUTDIR,
        help=f"Assist checkpoint outdir (default {DEFAULT_BH_ASSIST_OUTDIR})",
    )
    p.add_argument(
        "--continue-from",
        default="",
        help="BANNED for assist (guarded); do not pass hang bh_v1 zip",
    )
    p.add_argument("--train-steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-json", default="", help="Optional manifest JSON path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refuse_hang_ppo_continue(args.continue_from or None)
    out = refuse_bh_assist_output_path(args.output_dir)

    if args.dry_run:
        payload = dry_run_manifest(
            buffer_path=str(args.buffer).strip() or None,
            output_dir=out,
            n_train_steps=int(args.train_steps),
        )
        print(json.dumps(payload, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
        return 0

    buffer_path = str(args.buffer).strip()
    if buffer_path:
        rows, meta = load_assist_rows_from_buffer_path(buffer_path)
    else:
        arrays = synthetic_combat_buffer(n=32, n_episodes=4, seed=int(args.seed))
        rows = build_assist_rows_from_buffer(arrays)
        meta = {"buffer_source": "synthetic"}
    if not rows:
        raise SystemExit("no assist train rows from buffer")

    ckpt = train_linear_assist_ranker(
        rows, seed=int(args.seed), steps=int(args.train_steps)
    )
    manifest = {
        "protocol": PROTOCOL_ID,
        "dry_run": False,
        "output_dir": str(out),
        "buffer": buffer_path or None,
        "n_assist_rows": len(rows),
        "train_steps": int(args.train_steps),
        "hang_policy_swap": False,
        "model_kind": "linear_assist_ranker",
    }
    manifest.update({k: meta[k] for k in ("buffer_version", "n_transitions") if k in meta})
    ckpt_path = save_assist_checkpoint(out, ckpt, manifest)
    print(f"bh_assist checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
