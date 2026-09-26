#!/usr/bin/env python3
"""Export tip #2 combat critic .pt → ONNX and verify (Colab / local)."""
from __future__ import annotations

import argparse
import json
import sys

from sts2_env.colab.combat_critic import export_critic_onnx, verify_critic_onnx


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Colab combat critic ONNX export + verify")
    p.add_argument("ckpt", help="Input .pt from train_critic_smoke")
    p.add_argument("onnx", help="Output .onnx path")
    p.add_argument("--no-verify", action="store_true", help="Skip ORT parity check")
    args = p.parse_args(argv)
    out = export_critic_onnx(args.ckpt, args.onnx)
    payload: dict = {"onnx_path": out}
    if not args.no_verify:
        payload["verify"] = verify_critic_onnx(args.ckpt, out)
        print("ONNX_EXPORT_PASS", out, payload["verify"]["max_abs_err"])
    else:
        print("ONNX_EXPORT_OK", out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
