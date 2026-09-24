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

# Remaining planning rows after shard0/v0 (~44158 recorded).
_REMAINING = PLANNING_TARGET_ROWS - EMPIRICAL_PLANNING_ROWS_V0
RESUME_N_STEPS = combat_steps_for_planning_rows(_REMAINING)

RESUME_CLI = (
    "python scripts/collect_runenv_combat.py --planning-only --jev off "
    "--noncombat-policy ppo --combat-policy ppo "
    "--policy-zip /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip "
    f"--planning-shard shard01 --n-envs 8 --n-steps {RESUME_N_STEPS}"
)


def main() -> None:
    print("canonical_obs_size:", RUN_OBS_SIZE, "(181 combat obs_v1 + 20 run tail)")
    print("action_size:", TOTAL_ACTIONS)
    print("dir:", PLANNING_DIR)
    print("v0_backup_readonly:", PLANNING_V0_BACKUP_DIR)
    print("merged_shard0:", PLANNING_DEFAULT_OUT, f"(n≈{EMPIRICAL_PLANNING_ROWS_V0}; do not overwrite)")
    print("keys:", ", ".join(PLANNING_REQUIRED_KEYS))
    print("yield_v0:", EMPIRICAL_PLANNING_YIELD_V0, "planning_rows / combat_step")
    print(
        "formula: n_steps_total ≈ ceil(target_planning / yield); "
        f"500k planning ≈ {combat_steps_for_planning_rows(PLANNING_TARGET_ROWS)} combat steps (w=8 splits n_steps)"
    )
    print("combat_readonly:", COLAB_V1_COLLECT_OUT)
    print("resume_shard01:", RESUME_CLI)


if __name__ == "__main__":
    main()
