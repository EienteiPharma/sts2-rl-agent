#!/usr/bin/env python3
"""CLI wrapper for Colab combat feature extract (local smoke; no GPU required)."""
from __future__ import annotations

import argparse
import json
import sys

from sts2_env.colab.combat_transition_features import extract_combat_features


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export combat transitions feature table")
    p.add_argument("npz", help="Source transitions.npz")
    p.add_argument("out", help="Output .parquet or .npz")
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--sample-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    meta = extract_combat_features(
        args.npz,
        args.out,
        min_rows=args.min_rows,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
