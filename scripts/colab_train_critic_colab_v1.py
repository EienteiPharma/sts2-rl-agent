#!/usr/bin/env python3
"""Formal Colab v1 critic train + ONNX export (full buffer; warm-start smoke .pt)."""
from __future__ import annotations

import argparse
import json
import sys

from sts2_env.colab.combat_critic import (
    CriticColabV1TrainConfig,
    DEFAULT_ONNX_VERIFY_COLAB_V1,
    export_critic_onnx,
    train_critic_colab_v1,
    verify_critic_onnx,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Colab v1 formal critic train + ONNX")
    p.add_argument("features", help="Full colab_v1 transitions.npz")
    p.add_argument("init_ckpt", help="Warm-start combat_critic_smoke.pt")
    p.add_argument("out_ckpt", help="Output combat_critic_colab_v1.pt")
    p.add_argument("out_onnx", help="Output combat_critic_colab_v1.onnx")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    meta = train_critic_colab_v1(
        args.features,
        args.out_ckpt,
        args.init_ckpt,
        config=CriticColabV1TrainConfig(
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
    )
    onnx_out = export_critic_onnx(args.out_ckpt, args.out_onnx)
    verify = verify_critic_onnx(
        args.out_ckpt,
        onnx_out,
        max_abs_err=DEFAULT_ONNX_VERIFY_COLAB_V1,
    )
    print("CRITIC_COLAB_V1_PASS", meta["ckpt_path"], verify["max_abs_err"])
    print(json.dumps({"train": meta, "onnx": verify}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
