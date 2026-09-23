"""Combat turn-plan path (path A): semantic keys, full board, enumerated plans.

Jev will later Choice among ``plan_id`` values (``combat_turn_plan_choice``),
not stepwise ``combat_step_choice`` / opaque ``aN`` ids. Plans are
code-enumerated only; steps are semantic keys from the current legal set.

Optional advisory hints: ``bh_assist(board)`` → ranked semantics + risk notes
(``docs/BH_ASSIST_CONTRACT.md``); **not** the hang actor; execution stays
``bh_v1`` until turn-plan HOLD clears.

No live HTTP in this module. Default hang combat remains ``--combat-policy ppo``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import sts2_env.cards  # noqa: F401

from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PowerId
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    is_potion_action,
)

CHOICE_COMBAT_TURN_PLAN = "combat_turn_plan_choice"
SEMANTIC_END_TURN = "end_turn"
MAX_PLAN_STEPS = 8
MAX_CANDIDATE_PLANS = 512

COMBAT_TURN_PLAN_INSTRUCTIONS = (
    "Choose exactly one plan_id from the legal shortlist. Each plan is a "
    "fixed, code-enumerated sequence of semantic combat steps for this turn "
    "(not free-generated text). Do not invent plan_ids or steps outside the "
    "shortlist."
)

_TRACKED_POWERS = (
    PowerId.STRENGTH,
    PowerId.DEXTERITY,
    PowerId.VULNERABLE,
    PowerId.WEAK,
    PowerId.FRAIL,
    PowerId.ARTIFACT,
)


def _enum_name(obj: Any, *, attr: str | None = None) -> str:
    if obj is None:
        return "?"
    if attr is not None:
        obj = getattr(obj, attr, obj)
    return str(getattr(obj, "name", obj))


def _cost_text(card: Any) -> str:
    if getattr(card, "has_energy_cost_x", False):
        return "X"
    return str(getattr(card, "cost", "?"))


def _powers_map(creature: Creature) -> dict[str, int]:
    out: dict[str, int] = {}
    for pid in _TRACKED_POWERS:
        amt = int(creature.get_power_amount(pid) or 0)
        if amt:
            out[_enum_name(pid)] = amt
    return out


def _intent_line(combat: CombatState, enemy: Creature) -> str:
    ai = combat.enemy_ais.get(enemy.combat_id)
    if ai is None:
        return "unknown"
    intents = list(getattr(getattr(ai, "current_move", None), "intents", None) or [])
    if not intents:
        return "unknown"
    bits: list[str] = []
    for intent in intents:
        it = getattr(intent, "intent_type", None)
        name = _enum_name(it).lower()
        dmg = int(getattr(intent, "damage", 0) or 0)
        hits = int(getattr(intent, "hits", 1) or 1)
        if dmg > 0 and hits > 1:
            bits.append(f"{name} {dmg}x{hits}")
        elif dmg > 0:
            bits.append(f"{name} {dmg}")
        else:
            bits.append(name)
    return "/".join(bits) if bits else "unknown"


def _pile_summary(cards: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": _enum_name(getattr(c, "card_id", None)),
            "cost": _cost_text(c),
            "upgraded": bool(getattr(c, "upgraded", False)),
        }
        for c in cards
    ]


def semantic_key_for_gym_action(
    combat: CombatState,
    action: int,
    *,
    owner: Creature | None = None,
) -> str | None:
    """Map a gym combat action index to a stable semantic key (current state)."""
    idx = int(action)
    if idx == ACTION_END_TURN:
        return SEMANTIC_END_TURN
    acting = owner or combat.primary_player
    owner_state = combat.combat_player_state_for(acting)
    hand = owner_state.hand if owner_state is not None else combat.hand
    potions = owner_state.potions if owner_state is not None else combat.potions
    if is_potion_action(idx):
        slot, tgt = action_to_potion_and_target(idx)
        if slot is None:
            return None
        potion = potions[slot] if 0 <= slot < len(potions) else None
        pid = _enum_name(getattr(potion, "potion_id", None) if potion else None)
        if tgt is None:
            return f"potion:{pid}:s{slot}@self"
        return f"potion:{pid}:s{slot}@e{int(tgt)}"
    hand_i, tgt = action_to_card_and_target(idx)
    if hand_i is None:
        return None
    card = hand[hand_i] if 0 <= hand_i < len(hand) else None
    cid = _enum_name(getattr(card, "card_id", None) if card is not None else None)
    if tgt is None:
        return f"play:{cid}:h{hand_i}@self"
    return f"play:{cid}:h{hand_i}@e{int(tgt)}"


def gym_action_for_semantic_key(
    combat: CombatState,
    mask: np.ndarray,
    key: str,
    *,
    owner: Creature | None = None,
) -> int | None:
    """Resolve semantic key to gym index if legal on ``mask``; else None."""
    want = str(key).strip()
    if not want:
        return None
    mask_arr = np.asarray(mask)
    for action in np.flatnonzero(mask_arr == 1):
        if semantic_key_for_gym_action(combat, int(action), owner=owner) == want:
            return int(action)
    return None


def legal_semantic_keys(
    combat: CombatState,
    mask: np.ndarray,
    *,
    owner: Creature | None = None,
) -> tuple[str, ...]:
    """Sorted unique semantic keys for currently legal gym actions."""
    keys: set[str] = set()
    mask_arr = np.asarray(mask)
    for action in np.flatnonzero(mask_arr == 1):
        sk = semantic_key_for_gym_action(combat, int(action), owner=owner)
        if sk:
            keys.add(sk)
    return tuple(sorted(keys))


def serialize_combat_board_full(
    combat: CombatState,
    mask: np.ndarray,
    *,
    owner: Creature | None = None,
) -> dict[str, Any]:
    """Full combat board for turn-plan Jev (no hand/enemy truncation)."""
    acting = owner or combat.primary_player
    owner_state = combat.combat_player_state_for(acting)
    hand = list(owner_state.hand if owner_state is not None else combat.hand)
    potions = list(owner_state.potions if owner_state is not None else combat.potions)
    mask_arr = np.asarray(mask)
    end_turn_legal = (
        int(mask_arr[ACTION_END_TURN]) == 1 if mask_arr.size > ACTION_END_TURN else False
    )
    draw = list(combat.draw_pile)
    discard = list(combat.discard_pile)
    exhaust = list(combat.exhaust_pile)
    enemies_alive = [e for e in combat.enemies if e.is_alive]
    return {
        "self": {
            "hp": int(acting.current_hp),
            "max_hp": int(acting.max_hp),
            "block": int(acting.block),
            "energy": int(combat.current_energy),
            "max_energy": int(getattr(combat, "max_energy", combat.current_energy)),
            "powers": _powers_map(acting),
            "hand": [
                {
                    "name": _enum_name(getattr(c, "card_id", None)),
                    "cost": _cost_text(c),
                    "playable": bool(combat.can_play_card(c)),
                    "hand_index": i,
                }
                for i, c in enumerate(hand)
            ],
            "potions": [
                None
                if p is None
                else {"name": _enum_name(getattr(p, "potion_id", None)), "slot": i}
                for i, p in enumerate(potions)
            ],
        },
        "piles": {
            "draw": _pile_summary(draw),
            "discard": _pile_summary(discard),
            "exhaust": _pile_summary(exhaust),
            "draw_n": len(draw),
            "discard_n": len(discard),
            "exhaust_n": len(exhaust),
        },
        "enemies": [
            {
                "name": _enum_name(enemy, attr="monster_id"),
                "slot": i,
                "hp": int(enemy.current_hp),
                "max_hp": int(enemy.max_hp),
                "block": int(enemy.block),
                "intent": _intent_line(combat, enemy),
                "vuln": int(enemy.get_power_amount(PowerId.VULNERABLE) or 0),
                "weak": int(enemy.get_power_amount(PowerId.WEAK) or 0),
            }
            for i, enemy in enumerate(enemies_alive)
        ],
        "turn": {
            "player_turn_index": int(getattr(combat, "round_number", 1) or 1),
            "end_turn_legal": bool(end_turn_legal),
        },
    }


@dataclass(frozen=True)
class TurnPlanCandidate:
    plan_id: str
    steps: tuple[str, ...]


def _plan_criteria_summary(plan: TurnPlanCandidate) -> str:
    if not plan.steps:
        return "(empty)"
    return " → ".join(plan.steps)


def enumerate_candidate_plans(
    legal_keys: Sequence[str],
    *,
    max_steps: int = MAX_PLAN_STEPS,
    max_plans: int = MAX_CANDIDATE_PLANS,
) -> tuple[TurnPlanCandidate, ...]:
    """Deterministic bounded BFS over legal semantic keys (length ≤ max_steps).

    Prefixes that end with ``end_turn`` are not extended. Generation stops at
    ``max_plans`` candidates.
    """
    keys = tuple(sorted({str(k) for k in legal_keys if str(k).strip()}))
    if not keys or max_steps < 1 or max_plans < 1:
        return ()
    cap = int(max_steps)
    limit = int(max_plans)
    plans: list[TurnPlanCandidate] = []
    queue: list[tuple[str, ...]] = [(k,) for k in keys]
    head = 0
    while head < len(queue) and len(plans) < limit:
        prefix = queue[head]
        head += 1
        plans.append(TurnPlanCandidate(plan_id=f"plan_{len(plans):04d}", steps=prefix))
        if len(prefix) >= cap or prefix[-1] == SEMANTIC_END_TURN:
            continue
        for k in keys:
            queue.append(prefix + (k,))
    return tuple(plans)


def build_combat_turn_plan_choice_question(
    plans: Sequence[TurnPlanCandidate],
    *,
    instructions: str = COMBAT_TURN_PLAN_INSTRUCTIONS,
) -> dict[str, Any]:
    """Jev Choice payload: criteria keys are ``plan_id`` strings only."""
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {p.plan_id: _plan_criteria_summary(p) for p in plans},
    }


def jev_turn_plan_questions(
    board: dict[str, Any],
    plans: Sequence[TurnPlanCandidate],
) -> dict[str, dict[str, Any]]:
    """Stub-shaped question map for a future live adapter (offline-safe)."""
    del board  # state is passed separately to system_one; kept for call-site shape
    return {CHOICE_COMBAT_TURN_PLAN: build_combat_turn_plan_choice_question(plans)}


__all__ = [
    "CHOICE_COMBAT_TURN_PLAN",
    "COMBAT_TURN_PLAN_INSTRUCTIONS",
    "MAX_CANDIDATE_PLANS",
    "MAX_PLAN_STEPS",
    "SEMANTIC_END_TURN",
    "TurnPlanCandidate",
    "build_combat_turn_plan_choice_question",
    "enumerate_candidate_plans",
    "gym_action_for_semantic_key",
    "jev_turn_plan_questions",
    "legal_semantic_keys",
    "semantic_key_for_gym_action",
    "serialize_combat_board_full",
]
