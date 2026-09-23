"""loadout_v1 HOLD (elite+boss). Gate: overall ≥70% and Boss ≥40%.

Locked protocol: docs/HOLD_PROTOCOL.md

Hang 2026-09-22 table on ``bh_v1`` (n_eps=20): overall 74.2 / elite 98.9 / Boss 49.4.
That table used ``eval_combat_suite.py --suite loadout_v1`` with the LOCKED
per-fixture relics/potions (01 Shuriken+Fire/Attack, 02 Bag of Marbles+Explosive/Block,
03 Vajra+Strength/Flex). Do not homogenize those three. Default relics apply
**only** when a fixture omits the ``relics`` key.

Not a hang-protocol Act1 RunEnv eval. Dual gates still ≥70 / Boss≥40 **on this
protocol**. Hang zip stays ``bh_v1``. Buffer train stays frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

HOLD_OVERALL_MIN = 0.70
HOLD_BOSS_MIN = 0.40
HOLD_ENC_IDS = list(range(16, 22))  # elite 16-18, boss 19-21
HOLD_FIXTURE_STEMS = ("loadout_v1_01", "loadout_v1_02", "loadout_v1_03")
HOLD_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "scripts" / "fixtures" / "loadout_v1"
)
HOLD_SEED_BASE = 40000
HOLD_SEED_FORMULA = "40000+fix*1000+enc*100+ep"
# Fallback only when a fixture omits the relics key. LOCKED HOLD 01–03 each
# have their own relics/potions — never overwrite those with this tuple.
HOLD_DEFAULT_RELICS = ("BURNING_BLOOD", "SHURIKEN")
HANG_HOLD_TABLE = {
    "date": "2026-09-22",
    "zip": "combat_ppo_obs_v1_bh_v1",
    "n_eps": 20,
    "overall": 0.7417,
    "elite": 0.9889,
    "boss": 0.4944,
    "summary": "evals/obs_v1_bh_v1_loadout_v1_n20.summary.json",
}
HUNG_COMBAT_ZIP = (
    "/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip"
)


def bucket_for_enc(enc_id: int) -> str:
    if enc_id < 16:
        return "normal" if enc_id >= 4 else "weak"
    if enc_id < 19:
        return "elite"
    return "boss"


def hold_passes(
    summary: dict[str, Any],
    *,
    min_overall: float = HOLD_OVERALL_MIN,
    min_boss: float = HOLD_BOSS_MIN,
) -> bool:
    overall = float(summary.get("overall", {}).get("win_rate", 0.0))
    boss = float(summary.get("boss", {}).get("win_rate", 0.0))
    return overall >= min_overall and boss >= min_boss


def summarize_hold_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _sum(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"n": 0, "win_rate": 0.0}
        wins = sum(1 for r in subset if r.get("win"))
        return {"n": n, "win_rate": round(wins / n, 4)}

    return {
        "overall": _sum(rows),
        "elite": _sum([r for r in rows if r.get("bucket") == "elite"]),
        "boss": _sum([r for r in rows if r.get("bucket") == "boss"]),
        "gate": {"overall_min": HOLD_OVERALL_MIN, "boss_min": HOLD_BOSS_MIN},
    }


def load_hold_fixtures(fixture_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(fixture_dir) if fixture_dir is not None else HOLD_FIXTURE_DIR
    fixtures: list[dict[str, Any]] = []
    for stem in HOLD_FIXTURE_STEMS:
        path = root / f"{stem}.json"
        if not path.is_file():
            raise SystemExit(f"HOLD fixture missing: {path}")
        import json

        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"HOLD fixture is not an object: {path}")
        fixtures.append(data)
    return fixtures


def hold_jobs(n_eps: int = 1, fixture_dir: Path | None = None) -> list[dict[str, Any]]:
    """3 fixtures × elite+boss encounters × n_eps (hang HOLD layout)."""
    from sts2_env.encounters.act1 import ALL_ACT1_ENCOUNTERS

    fixtures = load_hold_fixtures(fixture_dir)
    jobs: list[dict[str, Any]] = []
    for fix_i, fx in enumerate(fixtures):
        for enc_id in HOLD_ENC_IDS:
            setup = ALL_ACT1_ENCOUNTERS[enc_id]
            for ep in range(int(n_eps)):
                jobs.append(
                    {
                        "fixture_index": fix_i,
                        "fixture": fx,
                        "enc_id": enc_id,
                        "encounter_setup": setup,
                        "bucket": bucket_for_enc(enc_id),
                        "ep": ep,
                        "seed": HOLD_SEED_BASE + fix_i * 1000 + enc_id * 100 + ep,
                    }
                )
    return jobs


def _train_combat_mod():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_combat.py"
    spec = importlib.util.spec_from_file_location("train_combat_hold", path)
    assert spec is not None and spec.loader is not None
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


def apply_hold_relics(
    spec: dict[str, Any],
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Default relics only when the fixture omitted the ``relics`` key.

    Never overwrite LOCKED fixture relics or potions.
    """
    source = fixture if fixture is not None else spec
    if "relics" in source:
        return spec
    spec["relics"] = list(HOLD_DEFAULT_RELICS)
    spec["relics_defaulted"] = True
    return spec


def _materialize(fx: dict[str, Any]) -> dict[str, Any]:
    mod = _train_combat_mod()
    spec = mod.materialize_fixture(fx, suite="loadout_v1")
    return apply_hold_relics(spec, fixture=fx)


def options_from_hold_fixture(fx: dict[str, Any]) -> dict[str, Any]:
    """Hang-era ``options_from_fixture`` plus HOLD relic default when omitted."""
    spec = _materialize(fx)
    options: dict[str, Any] = {
        "deck": spec["deck"],
        "hp": spec["hp"],
        "max_hp": spec["max_hp"],
        "relics": spec["relics"],
    }
    if spec.get("potions") is not None:
        options["potions"] = spec["potions"]
    return options


def hold_protocol_meta(*, n_eps: int) -> dict[str, Any]:
    return {
        "suite": "loadout_v1",
        "fixtures": list(HOLD_FIXTURE_STEMS),
        "enc_ids": list(HOLD_ENC_IDS),
        "seed_base": HOLD_SEED_BASE,
        "seed_formula": HOLD_SEED_FORMULA,
        "n_eps": int(n_eps),
        "n_jobs": 3 * len(HOLD_ENC_IDS) * int(n_eps),
        "default_relics": list(HOLD_DEFAULT_RELICS),
        "relics": (
            "LOCKED fixture relics/potions applied via reset options; "
            "omitted relics key only → HOLD_DEFAULT_RELICS (do not overwrite 01–03)"
        ),
        "gate": {"overall_min": HOLD_OVERALL_MIN, "boss_min": HOLD_BOSS_MIN},
        "hang_table": dict(HANG_HOLD_TABLE),
        "zip": "combat_ppo_obs_v1_bh_v1",
    }


def hold_job_key(job: dict[str, Any]) -> tuple[int, int, int]:
    return (int(job["fixture_index"]), int(job["enc_id"]), int(job["ep"]))


def split_hold_jobs(jobs: list[dict[str, Any]], n_workers: int) -> list[list[dict[str, Any]]]:
    """Round-robin shard HOLD jobs. Empty shards dropped. Default 1 worker = one shard."""
    n = max(1, int(n_workers))
    if not jobs:
        return []
    shards: list[list[dict[str, Any]]] = [[] for _ in range(min(n, len(jobs)))]
    for i, job in enumerate(jobs):
        shards[i % len(shards)].append(job)
    return shards


def run_hold_job_list(
    jobs: list[dict[str, Any]],
    predict_fn: Callable[[np.ndarray, np.ndarray], int],
    *,
    choose_fn: Callable[[Any, np.ndarray, np.ndarray], int] | None = None,
    max_steps: int = 400,
) -> list[dict[str, Any]]:
    """Run a list of HOLD jobs serially. Used by workers=1 and each parallel worker."""
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    rows: list[dict[str, Any]] = []
    for job in jobs:
        options = options_from_hold_fixture(job["fixture"])
        env = STS2CombatEnv(encounter_pool=[job["encounter_setup"]])
        obs, info = env.reset(seed=int(job["seed"]), options=options)
        done = False
        steps = 0
        reward = 0.0
        terminated = False
        truncated = False
        while not done and steps < max_steps:
            mask = info.get("action_mask")
            if mask is None:
                mask = env.action_masks()
            mask_arr = np.asarray(mask)
            if choose_fn is not None:
                action = int(choose_fn(env, obs, mask_arr))
            else:
                action = int(predict_fn(obs, mask_arr))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        env.close()
        rows.append(
            {
                "win": bool(terminated and reward > 0),
                "bucket": job["bucket"],
                "enc_id": job["enc_id"],
                "fixture_index": job["fixture_index"],
                "seed": job["seed"],
                "steps": steps,
            }
        )
    return rows


def compact_hold_job(job: dict[str, Any]) -> dict[str, Any]:
    """Pickle-safe HOLD job (no encounter_setup callable)."""
    return {
        "fixture_index": int(job["fixture_index"]),
        "enc_id": int(job["enc_id"]),
        "ep": int(job["ep"]),
        "seed": int(job["seed"]),
        "bucket": str(job["bucket"]),
        "fixture": dict(job["fixture"]),
    }


def expand_hold_job(job: dict[str, Any]) -> dict[str, Any]:
    from sts2_env.encounters.act1 import ALL_ACT1_ENCOUNTERS

    out = dict(job)
    enc_id = int(job["enc_id"])
    out["encounter_setup"] = ALL_ACT1_ENCOUNTERS[enc_id]
    out["bucket"] = str(job.get("bucket") or bucket_for_enc(enc_id))
    return out


def hold_eval_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level spawn target. Reloads MaskablePPO; shards TypeSafe keys for jev."""
    import os

    from sts2_env.eval.jev_keys import TYPESAFE_API_KEY_ENV, key_for_worker, load_typesafe_api_keys

    worker_id = int(payload.get("worker_id", 0))
    model_path = str(payload["model"])
    device = str(payload.get("device") or "cpu")
    combat_policy = str(payload.get("combat_policy") or "ppo")
    max_steps = int(payload.get("max_steps", 400))
    jobs = [expand_hold_job(j) for j in payload.get("jobs") or []]

    zip_path = Path(model_path)
    if not zip_path.is_file():
        raise SystemExit(f"model zip not found: {model_path}")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib is required to load MaskablePPO zips. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    model = MaskablePPO.load(str(zip_path), device=device)

    def predict_fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    choose_fn = None
    telemetry = None
    if combat_policy in ("jev", "jev-turn"):
        from sts2_env.eval.jev_client import build_jev_adapter

        keys = load_typesafe_api_keys()
        pinned = key_for_worker(keys, worker_id)
        if pinned:
            os.environ[TYPESAFE_API_KEY_ENV] = pinned
        adapter = build_jev_adapter(
            enabled=True,
            api_keys=keys or None,
            key_index=worker_id,
        )
        rng = np.random.RandomState(int(worker_id))
        if combat_policy == "jev":
            from sts2_env.eval.combat_jev import CombatJevTelemetry, choose_combat_step

            telemetry = CombatJevTelemetry()

            def choose_fn(env, obs, mask):
                combat = getattr(env, "combat", None)
                if combat is None:
                    return predict_fn(obs, mask)
                local, _shadow = choose_combat_step(
                    combat,
                    mask,
                    rng,
                    model,
                    adapter=adapter,
                    combat_obs=obs,
                    telemetry=telemetry,
                )
                return int(local)
        else:
            from sts2_env.eval.combat_jev import CombatJevTelemetry
            from sts2_env.eval.combat_turn_plan import choose_combat_turn_plan_action

            telemetry = CombatJevTelemetry()

            def choose_fn(env, obs, mask):
                combat = getattr(env, "combat", None)
                if combat is None:
                    return predict_fn(obs, mask)
                local, _shadow = choose_combat_turn_plan_action(
                    combat,
                    mask,
                    rng,
                    model,
                    adapter=adapter,
                    combat_obs=obs,
                    env=env,
                    telemetry=telemetry,
                )
                return int(local)

    rows = run_hold_job_list(
        jobs, predict_fn, choose_fn=choose_fn, max_steps=max_steps
    )
    tel_blob = telemetry.to_dict() if telemetry is not None else None
    return {"worker_id": worker_id, "rows": rows, "combat_jev": tel_blob}


def run_hold_jobs_parallel(
    jobs: list[dict[str, Any]],
    *,
    model_path: str,
    device: str = "cpu",
    combat_policy: str = "ppo",
    max_steps: int = 400,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Spawn ProcessPool (same as collect). workers=1 calls hold_eval_worker in-process."""
    shards = split_hold_jobs(jobs, workers)
    payloads: list[dict[str, Any]] = []
    for i, shard in enumerate(shards):
        payloads.append(
            {
                "worker_id": i,
                "model": str(model_path),
                "device": device,
                "combat_policy": str(combat_policy or "ppo"),
                "max_steps": int(max_steps),
                "jobs": [compact_hold_job(j) for j in shard],
            }
        )
    if not payloads:
        return [], None
    if len(payloads) == 1:
        results = [hold_eval_worker(payloads[0])]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(hold_eval_worker, payloads)
    rows: list[dict[str, Any]] = []
    tel = None
    for row in results:
        rows.extend(row.get("rows") or [])
        blob = row.get("combat_jev")
        if blob:
            from sts2_env.eval.combat_jev import CombatJevTelemetry

            piece = CombatJevTelemetry.from_dict(blob)
            if tel is None:
                tel = piece
            else:
                tel.merge(piece)
    rows.sort(key=lambda r: (int(r["fixture_index"]), int(r["enc_id"]), int(r["seed"])))
    return rows, (tel.to_dict() if tel is not None else None)


def run_hold_smoke(
    predict_fn: Callable[[np.ndarray, np.ndarray], int],
    *,
    n_eps: int = 1,
    fixture_dir: Path | None = None,
    max_steps: int = 400,
    choose_fn: Callable[[Any, np.ndarray, np.ndarray], int] | None = None,
    workers: int = 1,
    model_path: str | None = None,
    device: str = "cpu",
    combat_policy: str = "ppo",
) -> dict[str, Any]:
    """Run HOLD episodes with a predict(obs, mask)->action callable. No SB3 import.

    Applies fixture relics/potions through ``env.reset(..., options=...)`` (hang
    ``options_from_fixture`` path). Win = ``terminated and reward > 0``.

    Optional ``choose_fn(env, obs, mask)`` is used when the caller needs the
    live ``STS2CombatEnv`` (combat-Jev bypass). Default path stays predict_fn.

    ``workers`` default 1 is the historical serial loop. ``workers>1`` uses a
    spawn ProcessPool (MaskablePPO is not thread-safe); sentry smoke passes 8.
    Parallel workers reload ``model_path`` and, on experimental
    ``--combat-policy jev``, shard TypeSafe keys via ``key_for_worker``.
    """
    jobs = hold_jobs(n_eps=n_eps, fixture_dir=fixture_dir)
    n_workers = max(1, int(workers))
    tel_blob: dict[str, Any] | None = None
    if n_workers <= 1:
        rows = run_hold_job_list(
            jobs, predict_fn, choose_fn=choose_fn, max_steps=max_steps
        )
    else:
        if not model_path:
            raise SystemExit(
                "HOLD --workers > 1 needs --model zip (each spawn worker "
                "reloads MaskablePPO; predict_fn closures do not pickle)"
            )
        rows, tel_blob = run_hold_jobs_parallel(
            jobs,
            model_path=str(model_path),
            device=device,
            combat_policy=combat_policy,
            max_steps=max_steps,
            workers=n_workers,
        )
    summary = summarize_hold_rows(rows)
    summary["passed"] = hold_passes(summary)
    summary["n_eps"] = int(n_eps)
    summary["protocol"] = hold_protocol_meta(n_eps=n_eps)
    summary["workers"] = n_workers
    if tel_blob is not None:
        summary["combat_jev_telemetry"] = tel_blob
    return summary
