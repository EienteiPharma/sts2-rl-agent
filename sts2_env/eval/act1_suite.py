"""Frozen Act1 RunEnv eval suite constants.

Seeds 200000..200049 (50; ``--n`` extends the range, hang bar ``--n 100``).
Primary metric: act1_clear_rate (max act >= 1).
Protocol: docs/act1_runenv_eval_protocol.md

Never conflate with combat suite win rates. Do not retune here.
"""
from __future__ import annotations

SEED_START = 200000
SEED_COUNT = 50
SEEDS = list(range(SEED_START, SEED_START + SEED_COUNT))
PROTOCOL_ID = "act1_runenv_eval_protocol.md LOCKED 2026-09-22"
HUNG_COMBAT_ZIP = (
    "/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip"
)

JEV_SHADOW_SKIPPED = "skipped"
JEV_SHADOW_STUB = "stub"

__all__ = [
    "HUNG_COMBAT_ZIP",
    "JEV_SHADOW_SKIPPED",
    "JEV_SHADOW_STUB",
    "PROTOCOL_ID",
    "SEED_COUNT",
    "SEED_START",
    "SEEDS",
]
