"""Combat bh_assist advisory helper (track 2).

Design: ``docs/BH_ASSIST_CONTRACT.md``. Not hang policy; not ``combat_step_choice``.
Hang step execution stays ``bh_v1`` / ``--combat-policy ppo`` until turn-plan HOLD clears.
Train entry: ``scripts/train_bh_assist_from_buffer.py`` / ``bh_assist_train.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sts2_env.eval.bh_assist_train import (
    BH_ASSIST_CKPT_NAME,
    DEFAULT_BH_ASSIST_OUTDIR,
    refuse_bh_assist_output_path,
)
from sts2_env.gym_env.observation import OBS_SIZE

BH_ASSIST_CKPT_ENV = "STS2_BH_ASSIST_CKPT"
DEFAULT_BH_ASSIST_CKPT = (
    "/workspace/sts2-sim/output/combat_bh_assist_v1/bh_assist_ranker.npz"
)

_RANKER_CACHE: dict[str, dict[str, np.ndarray] | None] = {}


@dataclass(frozen=True)
class TurnPlanBhAssistConfig:
    """Eval / turn-plan Choice wiring (hints only; Jev still picks plan_id)."""

    enabled: bool = False
    ckpt_path: str | None = None


@dataclass(frozen=True)
class BhAssistResult:
    """Advisory output for Jev prompts / shadow logs."""

    ranked_semantic: tuple[str, ...]
    risk_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "ranked_semantic": list(self.ranked_semantic),
            "risk_notes": list(self.risk_notes),
        }


def resolve_bh_assist_ckpt_path(explicit: str | Path | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(Path(explicit).expanduser())
    env = (os.environ.get(BH_ASSIST_CKPT_ENV) or "").strip()
    if env:
        return str(Path(env).expanduser())
    return DEFAULT_BH_ASSIST_CKPT


def load_bh_assist_ranker(path: str | Path) -> dict[str, np.ndarray] | None:
    """Load assist ranker npz; cache misses and load errors as ``None`` (fail-open)."""
    key = str(Path(path).expanduser().resolve())
    if key in _RANKER_CACHE:
        return _RANKER_CACHE[key]
    try:
        ckpt_path = Path(path).expanduser()
        if not ckpt_path.is_file():
            _RANKER_CACHE[key] = None
            return None
        refuse_bh_assist_output_path(ckpt_path, what="bh_assist checkpoint read")
        with np.load(ckpt_path) as data:
            w = np.asarray(data["w"], dtype=np.float32).reshape(-1)
            obs_size = int(np.asarray(data["obs_size"]).reshape(-1)[0])
        if w.size != obs_size or w.size != OBS_SIZE:
            _RANKER_CACHE[key] = None
            return None
        loaded = {"w": w, "obs_size": np.array([obs_size], dtype=np.int64)}
        _RANKER_CACHE[key] = loaded
        return loaded
    except Exception:
        _RANKER_CACHE[key] = None
        return None


def _heuristic_rank(
    board: Mapping[str, Any], legal_semantic_keys: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from sts2_env.eval.combat_turn_plan import (
        SEMANTIC_END_TURN,
        TurnPlanCandidate,
        incoming_attack_damage_from_board,
        score_turn_plan_candidate,
    )

    keys = tuple(str(k) for k in legal_semantic_keys if str(k).strip())
    if not keys:
        return (), ()
    board_dict = dict(board) if isinstance(board, Mapping) else {}
    scored: list[tuple[tuple[int, int, int, tuple[str, ...]], str]] = []
    for key in keys:
        plan = TurnPlanCandidate(plan_id=f"hint_{key}", steps=(key,))
        if key != SEMANTIC_END_TURN:
            plan = TurnPlanCandidate(plan_id=f"hint_{key}", steps=(key, SEMANTIC_END_TURN))
        comp = score_turn_plan_candidate(plan, board_dict)
        scored.append((comp, key))
    scored.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[0][3]))
    ranked = tuple(k for _c, k in scored)
    incoming = incoming_attack_damage_from_board(board_dict)
    self_row = board_dict.get("self") or {}
    hp = int(self_row.get("hp") or 0)
    notes: list[str] = []
    if incoming > 0:
        notes.append(f"incoming_attack_{incoming}")
    if incoming >= hp > 0:
        notes.append("lethal_incoming_this_turn")
    if SEMANTIC_END_TURN in keys and incoming > 0 and hp > 0 and incoming < hp:
        notes.append("consider_block_before_end_turn")
    return ranked, tuple(notes)


def rank_semantic_keys_with_ranker(
    ranker: Mapping[str, np.ndarray],
    combat_obs: np.ndarray,
    legal_semantic_keys: Sequence[str],
    *,
    combat: Any,
    mask: np.ndarray,
    owner: Any | None = None,
    board: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Rank legal semantics: heuristic step scores + tiny ranker context (not gym action id).

    Previous apply path used ``dot(w, obs) - action_index*1e-6``; ``ACTION_END_TURN==0``
    always won ties → ``ranked_semantic[0]==end_turn`` whenever legal. Fixed by scoring
    each semantic via ``score_turn_plan_candidate`` (same family as turn-plan prune).
    """
    from sts2_env.eval.combat_turn_plan import (
        SEMANTIC_END_TURN,
        TurnPlanCandidate,
        composite_heuristic_score,
        score_turn_plan_candidate,
        serialize_combat_board_full,
    )

    w = np.asarray(ranker["w"], dtype=np.float32).reshape(-1)
    obs = np.asarray(combat_obs, dtype=np.float32).reshape(-1)
    if obs.size != w.size:
        raise ValueError("bh_assist ranker obs dim mismatch")
    base = float(np.dot(w, obs))
    if board is None:
        board = serialize_combat_board_full(combat, mask, owner=owner)
    board_dict = dict(board) if isinstance(board, Mapping) else {}
    keys = tuple(str(k) for k in legal_semantic_keys if str(k).strip())
    scored: list[tuple[float, str]] = []
    for key in keys:
        if key == SEMANTIC_END_TURN:
            plan = TurnPlanCandidate(plan_id=f"hint_{key}", steps=(key,))
        else:
            plan = TurnPlanCandidate(
                plan_id=f"hint_{key}", steps=(key, SEMANTIC_END_TURN)
            )
        comp = score_turn_plan_candidate(plan, board_dict)
        hscore = composite_heuristic_score(comp[:3])
        score = float(hscore) + base * 1e-9
        scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(k for _, k in scored)


def bh_assist(
    board: Mapping[str, Any],
    *,
    legal_semantic_keys: Sequence[str],
    context: Mapping[str, Any] | None = None,
) -> BhAssistResult:
    """Rank legal semantic keys and attach risk notes for Jev (hints only).

    Must not call MaskablePPO or choose the executed combat action.
    Every ``ranked_semantic`` entry must appear in ``legal_semantic_keys``.
    """
    del context
    keys = tuple(str(k) for k in legal_semantic_keys if str(k).strip())
    ranked, notes = _heuristic_rank(board, keys)
    ranked = tuple(k for k in ranked if k in keys)
    return BhAssistResult(ranked_semantic=ranked, risk_notes=notes)


def try_turn_plan_bh_assist(
    board: Mapping[str, Any],
    legal_semantic_keys: Sequence[str],
    *,
    config: TurnPlanBhAssistConfig | None,
    combat_obs: np.ndarray,
    combat: Any,
    mask: np.ndarray,
    owner: Any | None = None,
) -> BhAssistResult | None:
    """Inject assist hints only when enabled and checkpoint loads; else fail-open (``None``)."""
    if config is None or not config.enabled:
        return None
    ckpt = resolve_bh_assist_ckpt_path(config.ckpt_path)
    ranker = load_bh_assist_ranker(ckpt)
    if ranker is None:
        return None
    keys = tuple(str(k) for k in legal_semantic_keys if str(k).strip())
    if not keys:
        return None
    try:
        ranked = rank_semantic_keys_with_ranker(
            ranker,
            combat_obs,
            keys,
            combat=combat,
            mask=mask,
            owner=owner,
            board=board,
        )
        ranked = tuple(k for k in ranked if k in keys)
        if not ranked:
            ranked, notes = _heuristic_rank(board, keys)
        else:
            _ranked_h, notes = _heuristic_rank(board, keys)
        ranked = tuple(k for k in ranked if k in keys)
        return BhAssistResult(ranked_semantic=ranked, risk_notes=notes)
    except Exception:
        return None


def bh_assist_instruction_suffix(assist: BhAssistResult) -> str:
    parts: list[str] = []
    if assist.ranked_semantic:
        parts.append(
            "Advisory ranked_semantic (hints only, not binding): "
            + ", ".join(assist.ranked_semantic)
        )
    if assist.risk_notes:
        parts.append("Risk notes: " + "; ".join(assist.risk_notes))
    if not parts:
        return ""
    return " " + " ".join(parts)


def resolve_turn_plan_bh_assist_from_flags(
    *,
    bh_assist: str | None,
    bh_assist_ckpt: str | None,
) -> TurnPlanBhAssistConfig | None:
    mode = (bh_assist or "off").strip().lower()
    if mode not in ("off", "on"):
        raise ValueError(f"unknown bh_assist mode {bh_assist!r}")
    if mode == "off":
        return TurnPlanBhAssistConfig(enabled=False)
    return TurnPlanBhAssistConfig(
        enabled=True,
        ckpt_path=resolve_bh_assist_ckpt_path(bh_assist_ckpt),
    )


__all__ = [
    "BH_ASSIST_CKPT_ENV",
    "BH_ASSIST_CKPT_NAME",
    "BhAssistResult",
    "DEFAULT_BH_ASSIST_CKPT",
    "DEFAULT_BH_ASSIST_OUTDIR",
    "TurnPlanBhAssistConfig",
    "bh_assist",
    "bh_assist_instruction_suffix",
    "load_bh_assist_ranker",
    "rank_semantic_keys_with_ranker",
    "refuse_bh_assist_output_path",
    "resolve_bh_assist_ckpt_path",
    "resolve_turn_plan_bh_assist_from_flags",
    "try_turn_plan_bh_assist",
]
