#!/usr/bin/env python3
"""Planning colab_v1 schema + yield formula + shard resume CLI."""
from __future__ import annotations

from sts2_env.gym_env.combat_buffer import COLAB_V1_COLLECT_OUT
from sts2_env.gym_env.planning_buffer import (
    EMPIRICAL_PLANNING_ROWS_V0,
    EMPIRICAL_PLANNING_YIELD_V0,
    PLANNING_DEFAULT_OUT,
    PLANNING_DIR,
    PLANNING_REQUIRED_KEYS,
    PLANNING_TARGET_ROWS,
    PLANNING_V0_BACKUP_DIR,
    combat_steps_for_planning_rows,
)
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, TOTAL_ACTIONS

# v0 yield used fail-open random tags; remeasure after bh_v1 noncombat PPO before scaling n_steps.
_REMAINING = PLANNING_TARGET_ROWS - EMPIRICAL_PLANNING_ROWS_V0
_RESUME_N_STEPS_FAILOPEN_EST = combat_steps_for_planning_rows(_REMAINING)

RESUME_SMOKE_CLI = (
    "python scripts/collect_runenv_combat.py --planning-only --jev off "
    "--noncombat-policy ppo --combat-policy ppo "
    "--policy-zip /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip "
    "--planning-shard shard01 --n-envs 8 --n-steps 50000"
)

RESUME_CLI = (
    RESUME_SMOKE_CLI
    + "  # then: n_steps ≈ ceil(remaining_planning / measured_yield); "
    f"failopen-era est was {_RESUME_N_STEPS_FAILOPEN_EST} — do not use until remeasured"
)


def main() -> None:
    print("canonical_obs_size:", RUN_OBS_SIZE, "(181 combat obs_v1 + 20 run tail)")
    print("action_size:", TOTAL_ACTIONS)
    print("dir:", PLANNING_DIR)
    print("v0_backup_readonly:", PLANNING_V0_BACKUP_DIR)
    print("merged_shard0:", PLANNING_DEFAULT_OUT, f"(n≈{EMPIRICAL_PLANNING_ROWS_V0}; do not overwrite)")
    print("keys:", ", ".join(PLANNING_REQUIRED_KEYS))
    print(
        "yield_v0_failopen_era:",
        EMPIRICAL_PLANNING_YIELD_V0,
        "planning_rows/combat_step (noncombat_ppo_failopen_random; not for collect planning)",
    )
    print(
        "formula: n_steps_total ≈ ceil(target_planning / measured_yield); "
        f"failopen-era 500k planning ≈ {combat_steps_for_planning_rows(PLANNING_TARGET_ROWS)} combat steps — unverified post bh_v1 PPO"
    )
    print("combat_readonly:", COLAB_V1_COLLECT_OUT)
    print("resume_smoke_shard01:", RESUME_SMOKE_CLI)
    print("resume_shard01_note:", RESUME_CLI)


if __name__ == "__main__":
    main()
