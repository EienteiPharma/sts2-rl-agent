#!/usr/bin/env python3
"""Slice HOLD turn-replay JSONL into a Boss/fail offline pack."""
from __future__ import annotations

import argparse
import json
import sys

from sts2_env.eval.hold_replay_pack import (
    DEFAULT_BOSS_FAIL_PACK_DIR,
    pack_hold_turn_replay_files,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pack hold_turn_replay JSONL (offline)")
    p.add_argument(
        "inputs",
        nargs="+",
        help="One or more hold_turn_replay_*.jsonl files",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_BOSS_FAIL_PACK_DIR,
        help=f"Output pack directory (default {DEFAULT_BOSS_FAIL_PACK_DIR})",
    )
    p.add_argument(
        "--filter",
        choices=("boss_fail", "boss_all", "fail_all", "retain_writer"),
        default="boss_fail",
        help=(
            "boss_fail=bucket boss and not win (default); boss_all=all boss; "
            "fail_all=all losses; retain_writer=same as replay writer retain rule"
        ),
    )
    args = p.parse_args(argv)
    manifest = pack_hold_turn_replay_files(
        args.inputs,
        out_dir=args.out_dir,
        filter_mode=args.filter,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
