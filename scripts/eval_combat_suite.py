#!/usr/bin/env python3
"""Combat suite eval. Hang HOLD: ``--suite loadout_v1 --n-eps 20``.

Protocol: ``docs/HOLD_PROTOCOL.md``. Dual gate overall ≥70 / Boss ≥40 on this
aligned protocol. Hang zip stays ``bh_v1``. Not Act1 RunEnv.

Box ops used to keep a bare Act1 22-enc copy at ``/workspace/sts2-sim/eval_combat_suite.py``
(no ``--suite loadout_v1``). That is **not** the hang table. Use this script.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sts2_env.eval.combat_hold import (
    HOLD_BOSS_MIN,
    HOLD_OVERALL_MIN,
    HUNG_COMBAT_ZIP,
    hold_protocol_meta,
    run_hold_smoke,
)
from sts2_env.gym_env.observation import OBS_SIZE

PROTOCOL_ID = "loadout_v1 HOLD LOCKED 2026-09-23"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combat suite eval (hang HOLD = --suite loadout_v1)"
    )
    parser.add_argument(
        "--suite",
        default="loadout_v1",
        help="Only loadout_v1 is the hang HOLD protocol (fixtures 01-03, enc 16-21)",
    )
    parser.add_argument(
        "--n-eps",
        type=int,
        default=20,
        help="Episodes per fixture×encounter (hang table used 20 → 360 fights)",
    )
    parser.add_argument(
        "--model",
        default=HUNG_COMBAT_ZIP,
        help="Combat MaskablePPO zip (obs_v1=181). Hang zip is bh_v1.",
    )
    parser.add_argument("--out", type=str, default="", help="Write summary JSON")
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 device: cpu (default, hang-shaped) | auto | cuda | cuda:N",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Gym steps per fight (hold smoke default)",
    )
    return parser.parse_args(argv)


def resolve_device(spec: str, *, cuda_available: bool | None = None) -> str:
    s = (spec or "cpu").strip().lower()
    if s == "cpu":
        return "cpu"
    if s == "auto":
        if cuda_available is None:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
        return "cuda" if cuda_available else "cpu"
    if s == "cuda" or s.startswith("cuda:"):
        if cuda_available is None:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
        if not cuda_available:
            raise SystemExit(f"--device {spec} requested but CUDA unavailable")
        return s
    raise SystemExit(f"unknown --device {spec!r} (use auto|cuda|cpu|cuda:N)")


def load_maskable_ppo(path: str, *, device: str) -> Any:
    zip_path = Path(path)
    if not zip_path.is_file():
        raise SystemExit(f"model zip not found: {path}")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib is required to load MaskablePPO zips. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    return MaskablePPO.load(str(zip_path), device=device)


def model_obs_dim(model: Any) -> int:
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def pct(rate: float) -> str:
    return f"{100.0 * float(rate):.1f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = str(args.suite).strip()
    if suite != "loadout_v1":
        raise SystemExit(
            f"unknown --suite {suite!r}. Hang HOLD is --suite loadout_v1 "
            "(fixtures 01-03, enc 16-21, relics/potions applied). "
            "Bare Act1 22-enc is a different protocol and is not this table."
        )
    n_eps = int(args.n_eps)
    if n_eps < 1:
        raise SystemExit("--n-eps must be >= 1")
    device = resolve_device(args.device)
    model = load_maskable_ppo(args.model, device=device)
    obs_dim = model_obs_dim(model)
    if obs_dim != OBS_SIZE:
        raise SystemExit(
            f"combat zip obs_dim={obs_dim} != OBS_SIZE={OBS_SIZE} (obs_v1)"
        )

    def predict_fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    summary = run_hold_smoke(predict_fn, n_eps=n_eps, max_steps=int(args.max_steps))
    overall = float(summary["overall"]["win_rate"])
    elite = float(summary["elite"]["win_rate"])
    boss = float(summary["boss"]["win_rate"])
    payload = {
        "protocol": PROTOCOL_ID,
        "suite": "loadout_v1",
        "model": str(Path(args.model).resolve()),
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **hold_protocol_meta(n_eps=n_eps),
        "overall": summary["overall"],
        "elite": summary["elite"],
        "boss": summary["boss"],
        "gate": summary["gate"],
        "passed": bool(summary["passed"]),
        "n_eps": n_eps,
    }
    text = (
        f"HOLD loadout_v1 n_eps={n_eps}  "
        f"{pct(overall)} / {pct(elite)} / Boss {pct(boss)}  "
        f"passed={payload['passed']} "
        f"(gate {HOLD_OVERALL_MIN:.0%}/{HOLD_BOSS_MIN:.0%})"
    )
    print(text)
    hang = payload["hang_table"]
    print(
        "  hang 2026-09-22 bh_v1 table: "
        f"{pct(hang['overall'])} / {pct(hang['elite'])} / Boss {pct(hang['boss'])} "
        "(n_eps=20; tolerance ~±3–5pp)"
    )
    out = str(args.out).strip()
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
