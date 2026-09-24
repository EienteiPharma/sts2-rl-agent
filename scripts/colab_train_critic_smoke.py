#!/usr/bin/env python3
"""Smoke-train combat critic from tip #1 features (Colab / local CPU)."""
from __future__ import annotations

import argparse
import json
import sys

from sts2_env.colab.combat_critic import CriticTrainConfig, train_critic_smoke


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Colab combat critic smoke train")
    p.add_argument("features", help="transitions.npz or tip#1 features.npz/.parquet")
    p.add_argument("ckpt", help="Output .pt checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    meta = train_critic_smoke(
        args.features,
        args.ckpt,
        config=CriticTrainConfig(
            epochs=args.epochs,
            min_rows=args.min_rows,
            seed=args.seed,
        ),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
