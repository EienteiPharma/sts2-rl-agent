"""Non-combat planning transitions from colab_v1 collect (Jev off, no TypeSafe).

Schema (npz): ``obs`` (N,201) run obs (181 combat obs_v1 + 20 run tail), ``next_obs``,
``action`` (RunEnv 157),
``reward``, ``done``, ``action_mask`` (N,157), ``phase_code`` (uint8),
``policy_tag`` (N,) unicode. Emitted only from ``RunEnvOnPolicyCombatEnv``
auto-noncombat steps (MAP/REST/CARD/…); combat ``transitions.npz`` unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.gym_env.run_env import RUN_OBS_SIZE, TOTAL_ACTIONS
from sts2_env.run.run_manager import RunManager

PLANNING_BUFFER_VERSION = 1
PLANNING_DIR = "/workspace/sts2-sim/output/runenv_planning_buffer_colab_v1"
PLANNING_V0_BACKUP_DIR = "/workspace/sts2-sim/output/runenv_planning_buffer_colab_v0"
PLANNING_DEFAULT_OUT = f"{PLANNING_DIR}/planning_transitions.npz"
PLANNING_DEFAULT_JSONL = f"{PLANNING_DIR}/planning_steps.jsonl"
PLANNING_SHARDS_SUBDIR = "shards"

# Empirical shard0 (500k combat steps, w=8): planning_v0 n≈44158 — fail-open random era; not training-grade alone.
EMPIRICAL_COMBAT_STEPS_V0 = 500_000
EMPIRICAL_PLANNING_ROWS_V0 = 44_158
EMPIRICAL_PLANNING_YIELD_V0 = EMPIRICAL_PLANNING_ROWS_V0 / EMPIRICAL_COMBAT_STEPS_V0
PLANNING_TARGET_ROWS = 500_000

_FROZEN_OUTDIRS = (
    "combat_ppo_obs_v1_bh_v1",
    "combat_runenv_onpolicy_v1",
    "combat_runenv_antiforget_v1",
)
_PROTECTED_COMBAT_COLLECT_DIRS = (
    "runenv_combat_buffer_ep",
    "runenv_combat_buffer_colab_v1",
)
_PROTECTED_PLANNING_DIRS = (
    "runenv_planning_buffer_colab_v0",
)


def _meta_json_path(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().with_suffix(".meta.json")


def planning_shard_path(shard_id: str, *, base_dir: str | Path = PLANNING_DIR) -> Path:
    """New shard file under ``…/shards/planning_{shard_id}.npz`` (append-safe)."""
    sid = str(shard_id).strip().replace("/", "_")
    if not sid:
        raise ValueError("planning shard id required")
    return Path(base_dir).expanduser() / PLANNING_SHARDS_SUBDIR / f"planning_{sid}.npz"


def combat_steps_for_planning_rows(
    target_planning_rows: int,
    *,
    yield_rate: float = EMPIRICAL_PLANNING_YIELD_V0,
) -> int:
    """``n_steps`` (total combat transitions, all workers) for ``target`` planning rows."""
    if yield_rate <= 0:
        raise ValueError("yield_rate must be > 0")
    return int(np.ceil(int(target_planning_rows) / float(yield_rate)))


def refuse_planning_path(
    path: str | Path,
    *,
    what: str = "planning buffer",
    allow_existing_merge: bool = False,
) -> Path:
    """Never write into bh_v1 / protected full combat buffers."""
    out = Path(path).expanduser()
    parts = set(out.parts)
    for name in _PROTECTED_PLANNING_DIRS:
        if name in parts:
            raise SystemExit(
                f"refusing to overwrite protected planning backup {what} {out}"
            )
    if (
        not allow_existing_merge
        and out.resolve() == Path(PLANNING_DEFAULT_OUT).expanduser().resolve()
        and out.is_file()
        and out.stat().st_size > 0
    ):
        raise SystemExit(
            f"refusing to overwrite merged planning {out} (shard0/v0 backed up); "
            f"use --planning-shard ID → {PLANNING_DIR}/{PLANNING_SHARDS_SUBDIR}/planning_<ID>.npz"
        )
    for name in _FROZEN_OUTDIRS:
        if out.name == name or name in parts:
            raise SystemExit(
                f"refusing to overwrite frozen {what} {out}; pick a new outdir"
            )
    for name in _PROTECTED_COMBAT_COLLECT_DIRS:
        if name in parts and out.name in ("transitions.npz", "planning_transitions.npz"):
            raise SystemExit(
                f"refusing to overwrite protected combat collect {what} {out}"
            )
    return out
PLANNING_REQUIRED_KEYS = (
    "obs",
    "next_obs",
    "action",
    "reward",
    "done",
    "action_mask",
    "phase_code",
    "policy_tag",
)

_PHASE_TO_CODE: dict[str, int] = {
    RunManager.PHASE_MAP_CHOICE: 1,
    RunManager.PHASE_REST_SITE: 2,
    RunManager.PHASE_CARD_REWARD: 3,
    RunManager.PHASE_BOSS_RELIC: 4,
    RunManager.PHASE_SHOP: 5,
    RunManager.PHASE_EVENT: 6,
    RunManager.PHASE_TREASURE: 7,
    RunManager.PHASE_COMBAT: 8,
}
_CODE_TO_PHASE = {v: k for k, v in _PHASE_TO_CODE.items()}

PLANNING_RECORD_PHASES = frozenset(
    {
        RunManager.PHASE_MAP_CHOICE,
        RunManager.PHASE_REST_SITE,
        RunManager.PHASE_CARD_REWARD,
        RunManager.PHASE_BOSS_RELIC,
        RunManager.PHASE_SHOP,
        RunManager.PHASE_EVENT,
        RunManager.PHASE_TREASURE,
    }
)


def phase_to_code(phase: str | None) -> int:
    if phase is None:
        return 0
    return int(_PHASE_TO_CODE.get(str(phase), 0))


def validate_planning_buffer(arrays: dict[str, np.ndarray]) -> int:
    missing = [k for k in PLANNING_REQUIRED_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"planning buffer missing keys: {missing}")
    n = int(np.asarray(arrays["obs"]).shape[0])
    if n == 0:
        raise ValueError("planning buffer is empty")
    obs = np.asarray(arrays["obs"])
    if obs.shape != (n, RUN_OBS_SIZE):
        raise ValueError(f"obs shape {obs.shape} != (N, {RUN_OBS_SIZE})")
    mask = np.asarray(arrays["action_mask"])
    if mask.shape != (n, TOTAL_ACTIONS):
        raise ValueError(f"action_mask shape {mask.shape} != (N, {TOTAL_ACTIONS})")
    return n


def stack_planning_records(records: dict[str, list]) -> dict[str, np.ndarray]:
    if not records["obs"]:
        raise ValueError("no planning transitions collected")
    n = len(records["obs"])
    return {
        "obs": np.asarray(records["obs"], dtype=np.float32),
        "next_obs": np.asarray(records["next_obs"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.int64),
        "reward": np.asarray(records["reward"], dtype=np.float32),
        "done": np.asarray(records["done"], dtype=np.bool_),
        "action_mask": np.asarray(records["action_mask"], dtype=np.int8),
        "phase_code": np.asarray(records["phase_code"], dtype=np.uint8),
        "policy_tag": np.asarray(records["policy_tag"], dtype=str),
    }


def concat_planning_buffers(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no planning shards to concat")
    for part in parts:
        validate_planning_buffer(part)
    return {
        key: np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
        for key in PLANNING_REQUIRED_KEYS
    }


def save_planning_buffer(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> Path:
    out = refuse_planning_path(path, what="planning buffer")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = validate_planning_buffer(arrays)
    payload: dict[str, Any] = {
        "buffer_kind": "runenv_planning_colab_v1",
        "buffer_version": PLANNING_BUFFER_VERSION,
        "n_transitions": n,
        "obs_size": RUN_OBS_SIZE,
        "action_size": TOTAL_ACTIONS,
        "keys": list(PLANNING_REQUIRED_KEYS),
        "phase_codes": _CODE_TO_PHASE,
        "jev": "off",
        "typesafe": "off",
        "source": "collect_runenv_combat auto_noncombat",
        "note": "Combat-only npz cannot be backfilled; re-collect with --jev off",
    }
    payload.update(meta or {})
    np.savez_compressed(out, **{k: arrays[k] for k in PLANNING_REQUIRED_KEYS})
    _meta_json_path(out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return out


@dataclass
class PlanningRecordStats:
    auto_noncombat_steps: int = 0
    recorded: int = 0
    skipped_phase: int = 0
    step_errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "auto_noncombat_steps": int(self.auto_noncombat_steps),
            "recorded": int(self.recorded),
            "skipped_phase": int(self.skipped_phase),
            "step_errors": int(self.step_errors),
        }

    @property
    def yield_per_auto_step(self) -> float:
        if self.auto_noncombat_steps <= 0:
            return 0.0
        return float(self.recorded) / float(self.auto_noncombat_steps)


class PlanningStepRecorder:
    """In-memory non-combat step log (side channel; does not touch combat npz)."""

    def __init__(self) -> None:
        self._records: dict[str, list] = {k: [] for k in PLANNING_REQUIRED_KEYS}
        self.stats = PlanningRecordStats()

    def __len__(self) -> int:
        return len(self._records["obs"])

    def note_auto_noncombat_step(self) -> None:
        self.stats.auto_noncombat_steps += 1

    def note_step_error(self) -> None:
        self.stats.step_errors += 1

    def record_step(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        action_mask: np.ndarray,
        phase: str | None,
        policy_tag: str,
    ) -> None:
        if phase not in PLANNING_RECORD_PHASES:
            self.stats.skipped_phase += 1
            return
        self.stats.recorded += 1
        self._records["obs"].append(np.asarray(obs, dtype=np.float32))
        self._records["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        self._records["action"].append(int(action))
        self._records["reward"].append(float(reward))
        self._records["done"].append(bool(done))
        m = np.asarray(action_mask, dtype=np.int8)
        if m.shape[0] != TOTAL_ACTIONS:
            fixed = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
            n = min(TOTAL_ACTIONS, m.shape[0])
            fixed[:n] = m[:n]
            m = fixed
        self._records["action_mask"].append(m)
        self._records["phase_code"].append(np.uint8(phase_to_code(phase)))
        self._records["policy_tag"].append(str(policy_tag))

    def to_arrays(self) -> dict[str, np.ndarray] | None:
        if not self._records["obs"]:
            return None
        arrays = stack_planning_records(self._records)
        validate_planning_buffer(arrays)
        return arrays

    def append_jsonl_line(self, path: str | Path, row: dict[str, Any]) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "EMPIRICAL_PLANNING_YIELD_V0",
    "PLANNING_DEFAULT_JSONL",
    "PLANNING_DEFAULT_OUT",
    "PLANNING_DIR",
    "PLANNING_RECORD_PHASES",
    "PLANNING_REQUIRED_KEYS",
    "PLANNING_TARGET_ROWS",
    "PLANNING_V0_BACKUP_DIR",
    "PlanningRecordStats",
    "PlanningStepRecorder",
    "combat_steps_for_planning_rows",
    "concat_planning_buffers",
    "phase_to_code",
    "planning_shard_path",
    "refuse_planning_path",
    "save_planning_buffer",
    "stack_planning_records",
    "validate_planning_buffer",
]
