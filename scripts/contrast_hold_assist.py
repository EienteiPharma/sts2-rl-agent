#!/usr/bin/env python3
"""Print hang | A | B HOLD contrast + STOP from three summary JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_env.eval.hold_assist_contrast import (
    contrast_hang_a_b,
    format_contrast_table,
    load_hold_summary,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contrast hang vs jev-turn A vs assist B")
    p.add_argument("--hang", required=True, help="HOLD summary JSON (ppo/bh_v1)")
    p.add_argument("--a", required=True, help="Arm A summary (jev-turn assist off)")
    p.add_argument("--b", required=True, help="Arm B summary (jev-turn assist on)")
    p.add_argument("--json-out", default="", help="Optional write full contrast JSON")
    args = p.parse_args(argv)
    result = contrast_hang_a_b(
        load_hold_summary(args.hang),
        load_hold_summary(args.a),
        load_hold_summary(args.b),
    )
    print(format_contrast_table(result))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if result["hang_gap_stop"]["triggered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
