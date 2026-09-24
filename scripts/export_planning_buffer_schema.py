#!/usr/bin/env python3
"""Planning colab_v1 schema + Surplus one-liner (combat 500k buffer read-only)."""
from __future__ import annotations

from sts2_env.gym_env.combat_buffer import COLAB_V1_COLLECT_OUT
from sts2_env.gym_env.planning_buffer import (
    PLANNING_DEFAULT_OUT,
    PLANNING_DIR,
    PLANNING_REQUIRED_KEYS,
)

USAGE = (
    "python scripts/collect_runenv_combat.py --planning-only --jev off "
    "--noncombat-policy ppo --combat-policy ppo "
    "--policy-zip /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip "
    "--n-envs 8 --n-steps 500000"
)


def main() -> None:
    print("dir:", PLANNING_DIR)
    print("npz:", PLANNING_DEFAULT_OUT)
    print("schema:", ", ".join(PLANNING_REQUIRED_KEYS), "| obs 151 run | action/mask 157")
    print("combat_readonly:", COLAB_V1_COLLECT_OUT, "(n=500k; do not overwrite)")
    print("usage:", USAGE)


if __name__ == "__main__":
    main()
