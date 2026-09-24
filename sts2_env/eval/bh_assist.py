"""Combat bh_assist advisory helper (track 2).

Design: ``docs/BH_ASSIST_CONTRACT.md``. Not hang policy; not ``combat_step_choice``.
Hang step execution stays ``bh_v1`` / ``--combat-policy ppo`` until turn-plan HOLD clears.
Train entry: ``scripts/train_bh_assist_from_buffer.py`` / ``bh_assist_train.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sts2_env.eval.bh_assist_train import DEFAULT_BH_ASSIST_OUTDIR, refuse_bh_assist_output_path


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


__all__ = [
    "BhAssistResult",
    "DEFAULT_BH_ASSIST_OUTDIR",
    "bh_assist",
    "refuse_bh_assist_output_path",
]

