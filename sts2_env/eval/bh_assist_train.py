"""Offline train entry for combat ``bh_assist`` (advisory ranker, not hang PPO).

See ``docs/BH_ASSIST_CONTRACT.md``. Never writes ``bh_v1`` / frozen hang zips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE
from sts2_env.gym_env.combat_buffer import (
    HUNG_OUTDIR_NAME,
    load_combat_buffer,
    refuse_frozen_path,
    synthetic_combat_buffer,
)
from sts2_env.gym_env.observation import OBS_SIZE

DEFAULT_BH_ASSIST_OUTDIR = "output/combat_bh_assist_v1"
DEFAULT_BH_ASSIST_V3_OUTDIR = "output/combat_bh_assist_v3"
PROTECTED_ASSIST_OUTDIR_NAMES = (
    "combat_bh_assist_v1",
    "combat_bh_assist_v2",
)
BH_ASSIST_MANIFEST_NAME = "bh_assist_train_manifest.json"
BH_ASSIST_CKPT_NAME = "bh_assist_ranker.npz"
PROTOCOL_ID = "bh_assist_buffer_rank_v1 LOCKED 2026-09-24"
PROTOCOL_ID_V3 = "bh_assist_buffer_rank_v3 LOCKED 2026-09-24"
# Pre-train HOLD eval (assist-on vs assist-off): docs/BH_ASSIST_CONTRACT.md
# lock_eval_then_v3 — formal n_eps=20, workers=8; n_eps=5 diagnostic only (no promotion).
ASSIST_EVAL_MIN_OVERALL_PP = 3
ASSIST_EVAL_MIN_BOSS_PP = 2
ASSIST_EVAL_FORMAL_N_EPS = 20
ASSIST_EVAL_FORMAL_WORKERS = 8
MAX_LEGAL_FOR_RANK = 32


def refuse_bh_assist_output_path(path: str | Path, *, what: str = "bh_assist outdir") -> Path:
    """Refuse frozen hang outdirs and hang ``final_model.zip`` targets."""
    out = refuse_frozen_path(path, what=what)
    parts = set(out.parts)
    for protected in PROTECTED_ASSIST_OUTDIR_NAMES:
        if out.name == protected or protected in parts:
            raise SystemExit(
                f"refusing to write {what} into protected assist tree {out}; "
                f"use e.g. {DEFAULT_BH_ASSIST_V3_OUTDIR} for new train"
            )
    if HUNG_OUTDIR_NAME in parts and out.name == "final_model.zip":
        raise SystemExit(
            f"refusing to write assist artifact over hang zip {out}; "
            f"use {DEFAULT_BH_ASSIST_OUTDIR}"
        )
    if out.suffix == ".zip" and HUNG_OUTDIR_NAME in parts:
        raise SystemExit(
            f"refusing assist {what} under hang tree {out}; not a hang policy swap"
        )
    return out


def refuse_hang_ppo_continue(path: str | Path | None) -> None:
    if path is None:
        return
    raw = str(path).strip()
    if not raw:
        return
    if HUNG_OUTDIR_NAME in Path(raw).parts and "final_model.zip" in raw:
        raise SystemExit(
            "bh_assist train does not continue-from hang bh_v1 zip; "
            "assist ranker trains from buffer labels only"
        )


def _legal_indices(mask_row: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(mask_row) == 1)]


def derive_risk_notes(
    *,
    reward: float,
    done: bool,
    incoming_attack: int = 0,
    player_hp: int | None = None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if done and reward > 0:
        notes.append("terminal_win")
    elif done:
        notes.append("terminal_loss")
    if incoming_attack > 0 and player_hp is not None and incoming_attack >= player_hp:
        notes.append("lethal_incoming_this_turn")
    elif incoming_attack > 0:
        notes.append(f"incoming_attack_{incoming_attack}")
    return tuple(notes)


def ranking_target_for_expert(
    legal_semantic: Sequence[str], expert_semantic: str | None
) -> tuple[str, ...]:
    keys = tuple(str(k) for k in legal_semantic if str(k).strip())
    if not keys:
        return ()
    if expert_semantic and expert_semantic in keys:
        tail = tuple(sorted(k for k in keys if k != expert_semantic))
        return (expert_semantic,) + tail
    return tuple(sorted(keys))


@dataclass(frozen=True)
class AssistTrainRow:
    obs: np.ndarray
    legal_actions: tuple[int, ...]
    expert_action: int
    ranked_semantic: tuple[str, ...]
    risk_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "legal_actions": list(self.legal_actions),
            "expert_action": int(self.expert_action),
            "ranked_semantic": list(self.ranked_semantic),
            "risk_notes": list(self.risk_notes),
        }


def expert_semantic_proxy(expert_action: int, legal: Sequence[int]) -> str | None:
    """Buffer v1 proxy: end_turn semantic when expert matches; else None (no aN ids)."""
    if int(expert_action) == ACTION_END_TURN and ACTION_END_TURN in legal:
        return "end_turn"
    return None


def build_assist_rows_from_buffer(
    arrays: dict[str, np.ndarray],
) -> list[AssistTrainRow]:
    obs = np.asarray(arrays["obs"], dtype=np.float32)
    action = np.asarray(arrays["action"], dtype=np.int64)
    reward = np.asarray(arrays["reward"], dtype=np.float32)
    done = np.asarray(arrays["done"], dtype=np.bool_)
    mask = np.asarray(arrays["action_mask"], dtype=np.int8)
    n = int(obs.shape[0])
    rows: list[AssistTrainRow] = []
    for i in range(n):
        legal = tuple(_legal_indices(mask[i])[:MAX_LEGAL_FOR_RANK])
        if not legal:
            continue
        expert = int(action[i])
        expert_sem = expert_semantic_proxy(expert, legal)
        legal_sem = ("end_turn",) if expert_sem == "end_turn" else ()
        ranked = ranking_target_for_expert(legal_sem, expert_sem)
        notes = derive_risk_notes(
            reward=float(reward[i]),
            done=bool(done[i]),
        )
        rows.append(
            AssistTrainRow(
                obs=obs[i],
                legal_actions=legal,
                expert_action=expert,
                ranked_semantic=ranked,
                risk_notes=notes,
            )
        )
    return rows


def train_linear_assist_ranker(
    rows: Sequence[AssistTrainRow],
    *,
    seed: int = 0,
    steps: int = 64,
    lr: float = 0.05,
) -> dict[str, np.ndarray]:
    """Tiny linear ranker: obs -> score per legal slot (assist hints, not hang PPO)."""
    if not rows:
        raise ValueError("no assist train rows")
    rng = np.random.RandomState(int(seed))
    w = rng.randn(OBS_SIZE).astype(np.float32) * 0.01
    for _ in range(max(1, int(steps))):
        row = rows[int(rng.randint(0, len(rows)))]
        legal = row.legal_actions
        if row.expert_action not in legal:
            continue
        scores = np.array([float(np.dot(w, row.obs)) for _ in legal], dtype=np.float32)
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / exp_s.sum()
        target = legal.index(row.expert_action)
        grad = probs.copy()
        grad[target] -= 1.0
        for j, _act in enumerate(legal):
            w -= float(lr) * grad[j] * row.obs
    return {"w": w.astype(np.float32), "obs_size": np.array([OBS_SIZE], dtype=np.int64)}


def save_assist_checkpoint(out_dir: Path, ckpt: dict[str, np.ndarray], meta: dict[str, Any]) -> Path:
    refuse_bh_assist_output_path(out_dir, what="bh_assist outdir")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / BH_ASSIST_CKPT_NAME
    refuse_bh_assist_output_path(ckpt_path, what="bh_assist checkpoint")
    np.savez_compressed(ckpt_path, **ckpt)
    manifest = dict(meta)
    manifest["checkpoint"] = str(ckpt_path.name)
    manifest_path = out_dir / BH_ASSIST_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return ckpt_path


def load_assist_rows_from_buffer_path(buffer_path: str | Path) -> tuple[list[AssistTrainRow], dict[str, Any]]:
    arrays, meta = load_combat_buffer(buffer_path)
    rows = build_assist_rows_from_buffer(arrays)
    return rows, meta


def dry_run_manifest(
    *,
    buffer_path: str | None,
    output_dir: str | Path,
    n_train_steps: int,
    pack_dir: str | None = None,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    out = refuse_bh_assist_output_path(output_dir)
    if buffer_path and str(buffer_path).strip():
        rows, meta = load_assist_rows_from_buffer_path(buffer_path)
        source = "disk"
        n_trans = int(meta.get("n_transitions") or len(rows))
    else:
        arrays = synthetic_combat_buffer(n=24, n_episodes=4, seed=0)
        rows = build_assist_rows_from_buffer(arrays)
        meta = {}
        source = "synthetic"
        n_trans = int(arrays["obs"].shape[0])
    payload: dict[str, Any] = {
        "protocol": protocol_id or PROTOCOL_ID,
        "dry_run": True,
        "output_dir": str(out),
        "buffer": buffer_path or None,
        "buffer_source": source,
        "n_transitions": n_trans,
        "n_assist_rows": len(rows),
        "train_steps": int(n_train_steps),
        "hang_policy_swap": False,
        "target": "ranked_semantic_and_risk_notes_for_jev",
        "sample_row": rows[0].as_dict() if rows else None,
    }
    if pack_dir and str(pack_dir).strip():
        payload["source_pack_dir"] = str(Path(pack_dir).resolve())
    return payload


__all__ = [
    "BH_ASSIST_CKPT_NAME",
    "BH_ASSIST_MANIFEST_NAME",
    "DEFAULT_BH_ASSIST_OUTDIR",
    "DEFAULT_BH_ASSIST_V3_OUTDIR",
    "PROTECTED_ASSIST_OUTDIR_NAMES",
    "PROTOCOL_ID_V3",
    "AssistTrainRow",
    "ASSIST_EVAL_FORMAL_N_EPS",
    "ASSIST_EVAL_FORMAL_WORKERS",
    "ASSIST_EVAL_MIN_BOSS_PP",
    "ASSIST_EVAL_MIN_OVERALL_PP",
    "PROTOCOL_ID",
    "build_assist_rows_from_buffer",
    "dry_run_manifest",
    "expert_semantic_proxy",
    "load_assist_rows_from_buffer_path",
    "ranking_target_for_expert",
    "refuse_bh_assist_output_path",
    "refuse_hang_ppo_continue",
    "save_assist_checkpoint",
    "train_linear_assist_ranker",
]
