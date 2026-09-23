"""loadout_v1 HOLD (elite+boss). Gate: overall ≥70% and Boss ≥40%.

Locked protocol: docs/HOLD_PROTOCOL.md

Hang 2026-09-22 table on ``bh_v1`` (n_eps=20): overall 74.2 / elite 98.9 / Boss 49.4.
That table used ``eval_combat_suite.py --suite loadout_v1`` with fixture relics
(BURNING_BLOOD, SHURIKEN) and potions applied via reset options. A later
materialize path dropped those keys, so 0-step HOLD on the same zip collapsed
(39.4 / 78.9 / Boss 0.0). This module is the aligned runner.

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
# Hang-era Ironclad mid-act HOLD fixtures included starter + Shuriken. Applied
# when the JSON omits ``relics`` so box-stripped copies still match the table.
HOLD_DEFAULT_RELICS = ("BURNING_BLOOD", "SHURIKEN")
HANG_HOLD_TABLE = {
    "date": "2026-09-22",
    "zip": "combat_ppo_obs_v1_bh_v1",
    "n_eps": 20,
    "overall": 0.742,
    "elite": 0.989,
    "boss": 0.494,
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


def apply_hold_relics(spec: dict[str, Any]) -> dict[str, Any]:
    """Keep fixture relics when present; else hang reconstruction defaults."""
    relics = spec.get("relics")
    if relics:
        spec["relics"] = list(relics)
        return spec
    spec["relics"] = list(HOLD_DEFAULT_RELICS)
    spec["relics_defaulted"] = True
    return spec


def _materialize(fx: dict[str, Any]) -> dict[str, Any]:
    mod = _train_combat_mod()
    spec = mod.materialize_fixture(fx, suite="loadout_v1")
    return apply_hold_relics(spec)


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
        "relics": "fixture relics/potions applied via reset options; omitted relics → HOLD_DEFAULT_RELICS",
        "gate": {"overall_min": HOLD_OVERALL_MIN, "boss_min": HOLD_BOSS_MIN},
        "hang_table": dict(HANG_HOLD_TABLE),
        "zip": "combat_ppo_obs_v1_bh_v1",
    }


def run_hold_smoke(
    predict_fn: Callable[[np.ndarray, np.ndarray], int],
    *,
    n_eps: int = 1,
    fixture_dir: Path | None = None,
    max_steps: int = 400,
) -> dict[str, Any]:
    """Run HOLD episodes with a predict(obs, mask)->action callable. No SB3 import.

    Applies fixture relics/potions through ``env.reset(..., options=...)`` (hang
    ``options_from_fixture`` path). Win = ``terminated and reward > 0``.
    """
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    rows: list[dict[str, Any]] = []
    for job in hold_jobs(n_eps=n_eps, fixture_dir=fixture_dir):
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
            action = int(predict_fn(obs, np.asarray(mask)))
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
    summary = summarize_hold_rows(rows)
    summary["passed"] = hold_passes(summary)
    summary["n_eps"] = int(n_eps)
    summary["protocol"] = hold_protocol_meta(n_eps=n_eps)
    return summary
