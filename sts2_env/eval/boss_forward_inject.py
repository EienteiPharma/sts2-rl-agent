"""Forward-inject HOLD jobs from Boss/fail replay pack (seed reproducible; no rewind)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sts2_env.eval.combat_hold import (
    HOLD_SEED_BASE,
    HOLD_SEED_FORMULA,
    bucket_for_enc,
    expand_hold_job,
    load_hold_fixtures,
    options_from_hold_fixture,
    run_hold_job_list,
)
from sts2_env.eval.hold_replay_pack import (
    BOSS_FAIL_PACK_PROTOCOL,
    load_pack_episodes,
    load_pack_manifest,
)

BOSS_FORWARD_INJECT_PROTOCOL = "boss_forward_inject_v1"


def infer_ep_from_seed(seed: int, fixture_index: int, enc_id: int) -> int:
    return int(seed) - HOLD_SEED_BASE - int(fixture_index) * 1000 - int(enc_id) * 100


def hold_job_from_replay_episode(
    doc: Mapping[str, Any],
    fixtures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a HOLD reset job; does **not** restore mid-fight board from replay turns."""
    fix_i = int(doc["fixture_index"])
    enc_id = int(doc["enc_id"])
    seed = int(doc["seed"])
    if fix_i < 0 or fix_i >= len(fixtures):
        raise ValueError(f"fixture_index out of range: {fix_i}")
    ep_raw = doc.get("ep")
    ep = int(ep_raw) if ep_raw is not None else infer_ep_from_seed(seed, fix_i, enc_id)
    expected_seed = HOLD_SEED_BASE + fix_i * 1000 + enc_id * 100 + ep
    if expected_seed != seed:
        raise ValueError(
            f"seed {seed} != formula {HOLD_SEED_FORMULA} "
            f"(expected {expected_seed} for fix={fix_i} enc={enc_id} ep={ep})"
        )
    fx = fixtures[fix_i]
    return {
        "fixture_index": fix_i,
        "fixture": dict(fx),
        "enc_id": enc_id,
        "bucket": str(doc.get("bucket") or bucket_for_enc(enc_id)),
        "ep": ep,
        "seed": seed,
        "pack_episode_ref": {
            "win": bool(doc.get("win")),
            "steps": int(doc.get("steps") or 0),
            "protocol": doc.get("protocol"),
        },
    }


def build_inject_jobs_from_pack(
    pack_dir: str | Path,
    *,
    fixture_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_pack_manifest(pack_dir)
    episodes = load_pack_episodes(pack_dir)
    fixtures = load_hold_fixtures(fixture_dir)
    jobs = [hold_job_from_replay_episode(doc, fixtures) for doc in episodes]
    meta = {
        "protocol": BOSS_FORWARD_INJECT_PROTOCOL,
        "pack_protocol": manifest.get("protocol") or BOSS_FAIL_PACK_PROTOCOL,
        "pack_dir": str(Path(pack_dir).resolve()),
        "n_jobs": len(jobs),
        "seed_formula": HOLD_SEED_FORMULA,
        "seed_base": HOLD_SEED_BASE,
        "note": (
            "Jobs start fights via env.reset(seed, fixture options) — forward only; "
            "replay turn/terminal JSON is labels for assist v3, not engine rewind."
        ),
    }
    return jobs, meta


def load_inject_jobs_json(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rehydrate HOLD jobs from ``inject_jobs.json`` (fixtures loaded from disk)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"inject jobs file must be object: {path}")
    raw_jobs = data.get("jobs") or []
    fixtures = load_hold_fixtures()
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        fix_i = int(raw["fixture_index"])
        enc_id = int(raw["enc_id"])
        seed = int(raw["seed"])
        ep = int(raw.get("ep") if raw.get("ep") is not None else infer_ep_from_seed(seed, fix_i, enc_id))
        jobs.append(
            {
                "fixture_index": fix_i,
                "fixture": dict(fixtures[fix_i]),
                "enc_id": enc_id,
                "bucket": str(raw.get("bucket") or bucket_for_enc(enc_id)),
                "ep": ep,
                "seed": seed,
                "pack_episode_ref": dict(raw.get("pack_episode_ref") or {}),
            }
        )
    meta = {k: v for k, v in data.items() if k != "jobs"}
    return jobs, meta


def write_inject_jobs(
    path: str | Path,
    jobs: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(meta),
        "jobs": [
            {
                "fixture_index": int(j["fixture_index"]),
                "enc_id": int(j["enc_id"]),
                "ep": int(j["ep"]),
                "seed": int(j["seed"]),
                "bucket": str(j.get("bucket") or ""),
                "pack_episode_ref": dict(j.get("pack_episode_ref") or {}),
            }
            for j in jobs
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def smoke_forward_inject(
    jobs: Sequence[Mapping[str, Any]],
    *,
    n: int,
    max_steps: int = 400,
    predict_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """Run up to ``n`` inject jobs with hang-shaped predict only (no Jev / no TypeSafe)."""
    if n < 1:
        return []
    expanded = [expand_hold_job(dict(j)) for j in jobs[: int(n)]]
    if predict_fn is None:
        import numpy as np

        def predict_fn(obs, mask):  # type: ignore[no-redef]
            valid = np.flatnonzero(np.asarray(mask) == 1)
            return int(valid[0])

    return run_hold_job_list(expanded, predict_fn, max_steps=max_steps)


__all__ = [
    "BOSS_FORWARD_INJECT_PROTOCOL",
    "build_inject_jobs_from_pack",
    "load_inject_jobs_json",
    "hold_job_from_replay_episode",
    "infer_ep_from_seed",
    "smoke_forward_inject",
    "write_inject_jobs",
]
