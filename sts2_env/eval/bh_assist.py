"""Combat bh_assist advisory helper (track 2) — contract stub only.

Design: ``docs/BH_ASSIST_CONTRACT.md``. Not hang policy; not ``combat_step_choice``.
Hang step execution stays ``bh_v1`` / ``--combat-policy ppo`` until turn-plan HOLD clears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
    del board, legal_semantic_keys, context
    raise NotImplementedError(
        "bh_assist is design-only on tip④; see docs/BH_ASSIST_CONTRACT.md"
    )


__all__ = ["BhAssistResult", "bh_assist"]
