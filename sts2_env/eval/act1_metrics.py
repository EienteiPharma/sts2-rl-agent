"""Act1 RunEnv eval summary and report writer.

Report JSON shape is frozen for sentry. Do not invent metrics or retune
``act1_clear_rate`` (max act >= 1).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sts2_env.eval.act1_suite import PROTOCOL_ID, SEED_COUNT, SEED_START
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE


def summarize_act1_rows(rows: list[dict]) -> dict:
    n = len(rows)
    clears = [r for r in rows if r["act1_clear"]]
    floors = [r["floor"] for r in rows]
    return {
        "n": n,
        "act1_clear_rate": round(sum(r["act1_clear"] for r in rows) / n, 4) if n else 0.0,
        "full_run_win_rate": round(sum(r["full_run_win"] for r in rows) / n, 4) if n else 0.0,
        "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 4) if n else 0.0,
        "median_floor": float(np.median(floors)) if floors else 0.0,
        "mean_floor": round(float(np.mean(floors)), 3) if floors else 0.0,
        "mean_hp_on_clear": round(float(np.mean([r["hp"] for r in clears])), 2) if clears else None,
        "max_floor": int(max(floors)) if floors else 0,
        "map_lowhp_hard_n": int(sum(int(r.get("map_lowhp_hard_n") or 0) for r in rows)),
        "map_lowhp_hard_eps": int(sum(1 for r in rows if int(r.get("map_lowhp_hard_n") or 0) > 0)),
        "map_lowhp_soft_b_n": int(sum(int(r.get("map_lowhp_soft_b_n") or 0) for r in rows)),
        "map_lowhp_soft_b_eps": int(sum(1 for r in rows if int(r.get("map_lowhp_soft_b_n") or 0) > 0)),
        "event_jev_used_n": int(sum(int(r.get("event_jev_used_n") or 0) for r in rows)),
        "event_safe_fallback_n": int(sum(int(r.get("event_safe_fallback_n") or 0) for r in rows)),
        "event_low_conf_random_n": int(sum(int(r.get("event_low_conf_random_n") or 0) for r in rows)),
        "event_options_empty_n": int(sum(int(r.get("event_options_empty_n") or 0) for r in rows)),
        "event_off_random_n": int(sum(int(r.get("event_off_random_n") or 0) for r in rows)),
        "potion_or_relic_safe_n": int(sum(int(r.get("potion_or_relic_safe_n") or 0) for r in rows)),
        "potion_or_relic_random_n": int(sum(int(r.get("potion_or_relic_random_n") or 0) for r in rows)),
    }


_summarize = summarize_act1_rows


def build_report(
    *,
    policy: str,
    model_path: str,
    combat_model_path: str,
    rows: list[dict],
    elapsed_s: float,
    jev: str = "off",
    jev_event: str = "off",
    jev_phases: str = "map,rest,card",
    jev_neow: str | None = None,
    start_with_neow: bool = False,
    map_lowhp: str = "on",
    map_lowhp_hard: str = "off",
    map_lowhp_soft_b: str = "off",
    seed_count: int | None = None,
) -> dict:
    summary = _summarize(rows)
    n_seeds = seed_count if seed_count is not None else (summary.get("n") or SEED_COUNT)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_ID,
        "suite": "act1_runenv",
        "policy": policy,
        "model": model_path or None,
        "combat_model": combat_model_path or None,
        "jev": jev,
        "jev_event": jev_event,
        "jev_phases": jev_phases,
        "jev_neow": jev_neow if jev_neow is not None else "off",
        "start_with_neow": bool(start_with_neow),
        "map_lowhp": map_lowhp,
        "map_lowhp_hard": map_lowhp_hard,
        "map_lowhp_soft_b": map_lowhp_soft_b,
        "character": "Ironclad",
        "ascension": 0,
        "seeds": {
            "start": SEED_START,
            "count": n_seeds,
            "list_head": list(range(SEED_START, SEED_START + n_seeds))[:3],
            "list_tail": list(range(SEED_START, SEED_START + n_seeds))[-3:],
        },
        "run_obs_size": RUN_OBS_SIZE,
        "combat_obs_size": OBS_SIZE,
        "jev_shadow": {
            "mode": "on" if jev == "on" else "stub",
            "note": (
                "Combat never uses Jev. --jev off: legal random + stub logs. "
                "--jev on: Choice/Score with confidence>=0.65, hp_pressure "
                "rest/continue, MAP UNKNOWN defer (unknown_deferred at conf<0.80 "
                "when hp_pressure>=2), MAP low-HP v1 map_lowhp (default on: "
                "hp_pressure>=2 + shop/rest legal → uncertain/error resamples among safe nodes, "
                "reason map_lowhp_random), opt-in v2 map_lowhp_hard (default off; "
                "--map-lowhp-hard on, reason map_lowhp_hard, counted as map_lowhp_hard_n), "
                "opt-in soft-B danger avoidance (default off: --map-lowhp-soft-b on, reason map_lowhp_soft_b, "
                "counted as map_lowhp_soft_b_n), "
                "and card_fit assist on true pick_card; potion/relic PHASE_CARD_REWARD screens "
                "safe fallback to take reward (potion_or_relic_safe_fallback). EVENT is off unless "
                "--jev-event on (or --jev-phases lists event); pending EVENT "
                "choose/confirm maps to combat slots; uncertain/error EVENT falls back to safe option "
                "(event_safe_fallback). Shop stays legal random. "
                "Errors fall back to legal random. Not an Act1-clear gate."
            ),
        },
        "elapsed_s": round(elapsed_s, 1),
        "summary": summary,
        "rows": rows,
    }


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    slim = {k: v for k, v in report.items() if k != "rows"}
    summary_path = out.with_name(out.stem + ".summary.json") if out.suffix == ".json" else out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n")
    return summary_path


__all__ = [
    "_summarize",
    "build_report",
    "summarize_act1_rows",
    "write_report",
]
