"""Collect hang PPO combat buffer from Boss forward-inject HOLD jobs (no rewind)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sts2_env.eval.combat_hold import expand_hold_job, options_from_hold_fixture
from sts2_env.gym_env.combat_buffer import REQUIRED_KEYS, stack_records

BOSS_FAIL_INJECT_BUFFER_PROTOCOL = "boss_fail_inject_buffer_v1"
DEFAULT_BOSS_FAIL_INJECT_BUFFER = "output/boss_fail_inject_buffer_v1/transitions.npz"


def collect_transitions_for_hold_jobs(
    jobs: list[dict[str, Any]],
    predict_fn: Callable[[np.ndarray, np.ndarray], int],
    *,
    max_steps: int = 400,
) -> dict[str, np.ndarray]:
    """Roll hung PPO on each inject job; one fight per job (forward reset only)."""
    import sts2_env.cards  # noqa: F401
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    records: dict[str, list] = {k: [] for k in REQUIRED_KEYS}
    for job in jobs:
        expanded = expand_hold_job(dict(job))
        options = options_from_hold_fixture(job["fixture"])
        env = STS2CombatEnv(encounter_pool=[expanded["encounter_setup"]])
        obs, info = env.reset(seed=int(job["seed"]), options=options)
        done = False
        steps = 0
        while not done and steps < int(max_steps):
            mask = info.get("action_mask")
            if mask is None:
                mask = env.action_masks()
            mask_arr = np.asarray(mask)
            action = int(predict_fn(obs, mask_arr))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            records["obs"].append(np.asarray(obs, dtype=np.float32))
            records["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
            records["action"].append(action)
            records["reward"].append(float(reward))
            records["done"].append(done)
            records["action_mask"].append(np.asarray(mask_arr, dtype=np.int8))
            steps += 1
            obs = next_obs
        env.close()
    if not records["obs"]:
        raise ValueError("no transitions collected from inject jobs")
    return stack_records(records)


def dry_run_collect_manifest(
    *,
    inject_jobs_path: str | Path,
    n_jobs: int,
    max_steps: int,
    model_path: str,
    pack_dir: str | None = None,
    out_path: str | Path = DEFAULT_BOSS_FAIL_INJECT_BUFFER,
) -> dict[str, Any]:
    return {
        "protocol": BOSS_FAIL_INJECT_BUFFER_PROTOCOL,
        "dry_run": True,
        "inject_jobs": str(Path(inject_jobs_path).resolve()),
        "pack_dir": pack_dir,
        "n_jobs": int(n_jobs),
        "max_steps": int(max_steps),
        "model": model_path,
        "out": str(out_path),
        "policy": "ppo_hung_bh_v1",
        "note": "Forward inject only — no TypeSafe / no jev-turn / no engine rewind",
    }


def load_inject_jobs_file(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sts2_env.eval.boss_forward_inject import load_inject_jobs_json

    return load_inject_jobs_json(path)


def resolve_inject_jobs(
    *,
    inject_jobs_path: str | Path | None,
    pack_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if inject_jobs_path and str(inject_jobs_path).strip():
        jobs, meta = load_inject_jobs_file(inject_jobs_path)
        return jobs, meta
    if pack_dir and str(pack_dir).strip():
        from sts2_env.eval.boss_forward_inject import build_inject_jobs_from_pack

        return build_inject_jobs_from_pack(pack_dir)
    raise ValueError("need --inject-jobs or --pack-dir")


__all__ = [
    "BOSS_FAIL_INJECT_BUFFER_PROTOCOL",
    "DEFAULT_BOSS_FAIL_INJECT_BUFFER",
    "collect_transitions_for_hold_jobs",
    "dry_run_collect_manifest",
    "load_inject_jobs_file",
]
