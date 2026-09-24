"""Hang-protocol combat transition collection (offline collect).

Default collectors use hang Jev non-combat. Opt-in ``jev_enabled=False`` skips
TypeSafe and uses PPO/random non-combat (see ``collect_runenv_combat.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.gym_env.combat_buffer import (
    REQUIRED_KEYS,
    concat_buffers,
    hang_protocol_meta,
    load_combat_buffer,
    refuse_frozen_path,
    save_combat_buffer,
    stack_records,
)


@dataclass
class CollectWorkerConfig:
    jev_enabled: bool = True
    noncombat_policy: str = "jev"
    noncombat_model: str | None = None
    combat_policy: str = "random"
    combat_model: str | None = None


def split_worker_steps(n_steps: int, n_envs: int) -> list[int]:
    n_envs = max(1, int(n_envs))
    n_steps = max(0, int(n_steps))
    if n_steps == 0:
        return [0] * n_envs
    base, rem = divmod(n_steps, n_envs)
    return [base + (1 if i < rem else 0) for i in range(n_envs)]


def legal_random_action(mask, rng: np.random.RandomState) -> int:
    valid = np.flatnonzero(np.asarray(mask) == 1)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def _load_maskable_ppo(model_path: str):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit("collect --policy model requires sb3-contrib / torch") from e
    return MaskablePPO.load(str(model_path), device="cpu")


def collect_transitions(
    env,
    n_steps: int,
    *,
    rng: np.random.RandomState,
    select_fn,
    reset_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Roll a hang-protocol combat env; store combat steps only."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    seed = int(reset_seed) if reset_seed is not None else int(rng.randint(0, 2**31 - 1))
    obs, info = env.reset(seed=seed)
    records: dict[str, list] = {k: [] for k in REQUIRED_KEYS}
    empty = 0
    steps = 0
    while steps < n_steps:
        if info.get("combat_unreachable"):
            empty += 1
            if empty > 8:
                break
            seed = int(rng.randint(0, 2**31 - 1))
            obs, info = env.reset(seed=seed)
            continue
        mask = env.action_masks()
        action = int(select_fn(obs, mask))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        records["obs"].append(np.asarray(obs, dtype=np.float32))
        records["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        records["action"].append(action)
        records["reward"].append(float(reward))
        records["done"].append(done)
        records["action_mask"].append(np.asarray(mask, dtype=np.int8))
        steps += 1
        if done:
            seed = int(rng.randint(0, 2**31 - 1))
            obs, info = env.reset(seed=seed)
        else:
            obs = next_obs
    return stack_records(records)


def collect_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level multiprocessing target."""
    from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

    n_steps = int(payload["n_steps"])
    worker_id = int(payload.get("worker_id", 0))
    seed = int(payload.get("seed", 0))
    shard = Path(payload["shard"])
    max_steps = int(payload.get("max_steps", 2000))
    rng = np.random.RandomState(seed + worker_id * 100003)

    cfg = CollectWorkerConfig(
        jev_enabled=bool(payload.get("jev_enabled", True)),
        noncombat_policy=str(payload.get("noncombat_policy", "jev")),
        noncombat_model=payload.get("noncombat_model"),
        combat_policy=str(payload.get("combat_policy", payload.get("policy", "random"))),
        combat_model=payload.get("combat_model") or payload.get("model"),
    )

    if cfg.jev_enabled:
        from sts2_env.eval.jev import load_typesafe_api_keys

        load_typesafe_api_keys()

    noncombat_ppo = None
    if cfg.noncombat_policy == "ppo" and cfg.noncombat_model:
        if Path(cfg.noncombat_model).is_file():
            noncombat_ppo = _load_maskable_ppo(cfg.noncombat_model)

    env = RunEnvOnPolicyCombatEnv(
        max_steps=max_steps,
        seed_offset=seed + worker_id,
        jev_key_index=worker_id,
        jev_enabled=cfg.jev_enabled,
        noncombat_policy=cfg.noncombat_policy,  # type: ignore[arg-type]
        noncombat_ppo_model=noncombat_ppo,
    )

    combat_model = None
    if cfg.combat_policy in ("model", "ppo"):
        model_path = cfg.combat_model
        if not model_path or not Path(model_path).is_file():
            env.close()
            raise SystemExit(f"collect combat zip not found: {model_path}")
        combat_model = _load_maskable_ppo(model_path)

    def select_fn(obs, mask):
        if combat_model is None:
            return legal_random_action(mask, rng)
        action, _ = combat_model.predict(
            np.asarray(obs), action_masks=mask, deterministic=False
        )
        return int(action)

    try:
        arrays = collect_transitions(
            env,
            n_steps,
            rng=rng,
            select_fn=select_fn,
            reset_seed=seed + worker_id,
        )
    finally:
        env.close()

    meta = hang_protocol_meta()
    meta.update(
        {
            "worker_id": worker_id,
            "policy": cfg.combat_policy,
            "combat_policy": cfg.combat_policy,
            "noncombat_policy": cfg.noncombat_policy,
            "jev": "on" if cfg.jev_enabled else "off",
            "typesafe": "on" if cfg.jev_enabled else "off",
            "n_steps_requested": n_steps,
            "seed": seed,
        }
    )
    save_combat_buffer(shard, arrays, meta)
    return {
        "shard": str(shard),
        "n_transitions": int(arrays["obs"].shape[0]),
        "worker_id": worker_id,
    }


def collect_parallel(
    *,
    out_path: str | Path,
    n_steps: int,
    n_envs: int = 1,
    policy: str = "random",
    model: str | None = None,
    seed: int = 0,
    max_steps: int = 2000,
    jev_enabled: bool = True,
    noncombat_policy: str = "jev",
    noncombat_model: str | None = None,
    combat_policy: str | None = None,
    combat_model: str | None = None,
) -> dict[str, Any]:
    """Collect hang combat transitions, optionally across ``n_envs`` workers."""
    out = refuse_frozen_path(out_path, what="buffer")
    n_envs = max(1, int(n_envs))
    combat_policy = combat_policy or policy
    combat_model = combat_model or model

    if jev_enabled:
        from sts2_env.eval.jev import load_typesafe_api_keys, typesafe_key_pool_summary, warn_n_envs

        load_typesafe_api_keys()
        note = warn_n_envs(n_envs)
        pool_extra = typesafe_key_pool_summary()
    else:
        note = None
        pool_extra = {"typesafe_key_count": 0, "typesafe_key_pool": "off"}

    if note:
        print(note)
    quotas = [q for q in split_worker_steps(n_steps, n_envs) if q > 0]
    shard_dir = out.parent / f".{out.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for i, quota in enumerate(quotas):
        payloads.append(
            {
                "n_steps": quota,
                "worker_id": i,
                "seed": int(seed),
                "policy": combat_policy,
                "combat_policy": combat_policy,
                "combat_model": combat_model,
                "model": combat_model,
                "jev_enabled": jev_enabled,
                "noncombat_policy": noncombat_policy,
                "noncombat_model": noncombat_model,
                "shard": str(shard_dir / f"shard_{i:02d}.npz"),
                "max_steps": int(max_steps),
            }
        )
    if len(payloads) == 1:
        results = [collect_worker(payloads[0])]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(collect_worker, payloads)
    parts = []
    for row in results:
        arrays, _meta = load_combat_buffer(row["shard"])
        parts.append(arrays)
    merged = concat_buffers(parts)
    meta = hang_protocol_meta()
    meta.update(
        {
            "policy": combat_policy,
            "combat_policy": combat_policy,
            "noncombat_policy": noncombat_policy,
            "jev": "on" if jev_enabled else "off",
            "typesafe": "on" if jev_enabled else "off",
            "n_envs": len(payloads),
            "n_steps_requested": int(n_steps),
            "seed": int(seed),
            "shards": [r["shard"] for r in results],
            **pool_extra,
        }
    )
    save_combat_buffer(out, merged, meta)
    hang = hang_protocol_meta()
    if not jev_enabled:
        hang = {**hang, "jev": "off", "typesafe": "off"}
    return {
        "out": str(out),
        "n_transitions": int(merged["obs"].shape[0]),
        "n_envs": len(payloads),
        "policy": combat_policy,
        "jev": "on" if jev_enabled else "off",
        "typesafe": "on" if jev_enabled else "off",
        "hang": hang,
        **pool_extra,
    }


__all__ = [
    "CollectWorkerConfig",
    "collect_parallel",
    "collect_transitions",
    "collect_worker",
    "legal_random_action",
    "split_worker_steps",
]
