#!/usr/bin/env python3
"""Print planning colab_v1 buffer schema (no collect)."""
from __future__ import annotations

import json

from sts2_env.gym_env.planning_buffer import (
    PLANNING_DEFAULT_JSONL,
    PLANNING_DEFAULT_OUT,
    PLANNING_DIR,
    PLANNING_REQUIRED_KEYS,
)


def main() -> None:
    print(
        json.dumps(
            {
                "dir": PLANNING_DIR,
                "npz": PLANNING_DEFAULT_OUT,
                "jsonl_audit": PLANNING_DEFAULT_JSONL,
                "keys": list(PLANNING_REQUIRED_KEYS),
                "obs_size": 151,
                "action_size": 157,
                "collect": (
                    "python scripts/collect_runenv_combat.py --jev off "
                    "--no-planning  # omit flag to emit planning side-channel"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
