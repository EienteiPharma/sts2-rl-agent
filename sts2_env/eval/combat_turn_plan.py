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

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import sts2_env.cards  # noqa: F401

from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PowerId
from sts2_env.eval.jev_types import JevAnswer, JevError
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    is_potion_action,
)

CHOICE_COMBAT_TURN_PLAN = "combat_turn_plan_choice"
SEMANTIC_END_TURN = "end_turn"
MAX_PLAN_STEPS = 8
MAX_CANDIDATE_PLANS = 512
TYPESAFE_CHOICE_PLATFORM_MAX = 255
DEFAULT_TURN_PLAN_CHOICE_CANDIDATES = 32
TURN_PLAN_CHOICE_CAP_ENV = "STS2_TURN_PLAN_CHOICE_CAP"
# Back-compat alias for tests/docs referring to the product default.
MAX_TURN_PLAN_CHOICE_CANDIDATES = DEFAULT_TURN_PLAN_CHOICE_CANDIDATES
MAX_REPLANS_PER_PLAYER_TURN = 3

CATA_FAILOPEN_TIMEOUT = "timeout"
CATA_FAILOPEN_ERROR = "error"
CATA_FAILOPEN_EMPTY = "empty"
CATA_FAILOPEN_ILLEGAL_PLAN = "illegal_plan"
CATA_FAILOPEN_CAP = "cap_exceeded"
CATA_FAILOPEN_MISSING_HUNG_PPO = "missing_hung_ppo"

COMBAT_TURN_PLAN_SYSTEM_RULES = (
    "Survival first: block or prevent telegraphed enemy intent damage before "
    "greedy damage. Prefer enumerated plan_ids only."
)

COMBAT_TURN_PLAN_INSTRUCTIONS = (
    "Choose exactly one plan_id from the legal shortlist. Each plan is a "
    "fixed, code-enumerated sequence of semantic combat steps for this turn "
    "(not free-generated text). Do not invent plan_ids or steps outside the "
    "shortlist. Use the combat snapshot (HP, block, energy, enemy intents, "
    "incoming damage) to block lethal telegraphed attacks before greedy damage."
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


def _humanize_semantic_step(step: str, board: dict[str, Any] | None) -> str:
    key = str(step).strip()
    if key == SEMANTIC_END_TURN:
        return "End turn"
    m = _PLAY_STEP_RE.match(key)
    if m:
        card = m.group("card")
        hi = int(m.group("hi"))
        entry = _hand_entry(board or {}, hi) if board else None
        if entry is not None:
            card = str(entry.get("name") or card)
            cost = entry.get("cost", "?")
        else:
            cost = "?"
        target = "self"
        if "@e" in key:
            target = f"enemy slot {key.split('@e', 1)[-1]}"
        return f"Play {card} (cost {cost}) → {target}"
    if key.startswith("potion:"):
        return f"Potion step {key}"
    return key


def _plan_criteria_summary(
    plan: TurnPlanCandidate, *, board: dict[str, Any] | None = None
) -> str:
    if not plan.steps:
        return "(empty)"
    return " → ".join(_humanize_semantic_step(s, board) for s in plan.steps)


def summarize_board_for_turn_plan_choice(board: dict[str, Any]) -> dict[str, Any]:
    """Compact combat snapshot for turn-plan Choice (full board optional separately)."""
    self_row = board.get("self") or {}
    hand = self_row.get("hand") or []
    enemies = board.get("enemies") or []
    turn = board.get("turn") or {}
    piles = board.get("piles") or {}
    incoming = incoming_attack_damage_from_board(board)
    return {
        "player": {
            "hp": int(self_row.get("hp") or 0),
            "max_hp": int(self_row.get("max_hp") or 0),
            "block": int(self_row.get("block") or 0),
            "energy": int(self_row.get("energy") or 0),
            "powers": dict(self_row.get("powers") or {}),
        },
        "incoming_attack_damage": incoming,
        "end_turn_legal": bool(turn.get("end_turn_legal")),
        "hand": [
            {
                "hand_index": int(c.get("hand_index", i)),
                "name": str(c.get("name") or "?"),
                "cost": str(c.get("cost") or "?"),
                "playable": bool(c.get("playable", True)),
            }
            for i, c in enumerate(hand)
        ],
        "enemies": [
            {
                "slot": int(e.get("slot", i)),
                "name": str(e.get("name") or "?"),
                "hp": int(e.get("hp") or 0),
                "max_hp": int(e.get("max_hp") or 0),
                "block": int(e.get("block") or 0),
                "intent": str(e.get("intent") or "unknown"),
                "vuln": int(e.get("vuln") or 0),
                "weak": int(e.get("weak") or 0),
            }
            for i, e in enumerate(enemies)
        ],
        "piles_n": {
            "draw": int(piles.get("draw_n") or 0),
            "discard": int(piles.get("discard_n") or 0),
            "exhaust": int(piles.get("exhaust_n") or 0),
        },
    }


def _choice_instructions_with_board(board: dict[str, Any], instructions: str) -> str:
    snap = summarize_board_for_turn_plan_choice(board)
    player = snap["player"]
    parts = [
        instructions,
        (
            f"Snapshot: player {player['hp']}/{player['max_hp']} HP, "
            f"block {player['block']}, energy {player['energy']}, "
            f"incoming attack ~{snap['incoming_attack_damage']}."
        ),
    ]
    for enemy in snap["enemies"]:
        parts.append(
            f"Enemy {enemy['slot']} ({enemy['name']}): intent {enemy['intent']}, "
            f"hp {enemy['hp']}/{enemy['max_hp']}, block {enemy['block']}."
        )
    if snap["hand"]:
        hand_bits = [
            f"h{row['hand_index']} {row['name']} cost {row['cost']}"
            for row in snap["hand"]
        ]
        parts.append("Hand: " + "; ".join(hand_bits) + ".")
    return " ".join(parts)


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


_PLAY_STEP_RE = re.compile(r"^play:(?P<card>[^:]+):h(?P<hi>\d+)(?:@.*)?$")
_PLAN_STEP_POTION_RE = re.compile(
    r"^potion:(?P<pid>[^:]+):s(?P<slot>\d+)(?:@(?P<tgt>.*))?$"
)


def plan_step_action_signature(key: str) -> tuple[str, ...] | None:
    """Card/potion/end signature ignoring hand index or potion slot."""
    raw = str(key).strip()
    if not raw:
        return None
    if raw == SEMANTIC_END_TURN:
        return (SEMANTIC_END_TURN,)
    after = raw.split("@", 1)
    target = after[1] if len(after) > 1 else "self"
    m = _PLAY_STEP_RE.match(raw)
    if m:
        return ("play", m.group("card"), target)
    m = _PLAN_STEP_POTION_RE.match(raw)
    if m:
        return ("potion", m.group("pid"), target)
    return None


def remap_plan_step_semantic_key(
    key: str, legal_keys: set[str] | frozenset[str]
) -> str | None:
    """Resolve a planned step to a current legal key when only index/slot drifted."""
    want = str(key).strip()
    if not want:
        return None
    if want in legal_keys:
        return want
    sig = plan_step_action_signature(want)
    if sig is None:
        return None
    matches = sorted(
        lk for lk in legal_keys if plan_step_action_signature(lk) == sig
    )
    if matches:
        return matches[0]
    return None
_INTENT_PART_ATTACK_RE = re.compile(
    r"^(?:attack|multi_attack)\s+(\d+)(?:x(\d+))?$",
    re.IGNORECASE,
)


def _minimal_board_for_plan_scoring() -> dict[str, Any]:
    return {
        "self": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "hand": []},
        "enemies": [],
        "turn": {"end_turn_legal": True},
    }


def _hand_entry(board: dict[str, Any], hand_index: int) -> dict[str, Any] | None:
    hand = board.get("self", {}).get("hand") or []
    for entry in hand:
        if int(entry.get("hand_index", -1)) == hand_index:
            return entry
    if 0 <= hand_index < len(hand):
        row = hand[hand_index]
        return row if isinstance(row, dict) else None
    return None


def _parse_energy_cost(cost: str | int | None) -> int:
    if cost is None:
        return 1
    raw = str(cost).strip()
    if not raw or raw.upper() == "X":
        return 99
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _est_block_from_card_name(card_name: str) -> int:
    u = str(card_name or "").upper().replace(" ", "_")
    if "IMPENETRABLE" in u:
        return 12
    if "ENTRENCH" in u:
        return 6
    if "DEFEND" in u or "BARRICADE" in u or "GLACIER" in u:
        return 5
    if "ARMOR" in u or "SHIELD" in u or "BARRIER" in u:
        return 4
    return 0


def _est_damage_from_card_name(card_name: str) -> int:
    u = str(card_name or "").upper()
    if "STRIKE" in u:
        return 6
    if "HEAVY" in u or "BLUDGEON" in u:
        return 12
    if "BASH" in u:
        return 8
    return 2


def incoming_attack_damage_from_board(board: dict[str, Any]) -> int:
    total = 0
    for enemy in board.get("enemies") or []:
        intent = str(enemy.get("intent") or "").lower()
        for part in intent.split("/"):
            part = part.strip()
            if not part or part == "unknown":
                continue
            match = _INTENT_PART_ATTACK_RE.match(part)
            if match:
                total += int(match.group(1)) * int(match.group(2) or 1)
    return total


def _analyze_plan_steps(
    steps: Sequence[str], board: dict[str, Any]
) -> tuple[int, int, int]:
    """Return (estimated_block, estimated_damage, energy_spent) for plan prefix."""
    block_gain = 0
    damage = 0
    energy = 0
    for step in steps:
        if step == SEMANTIC_END_TURN:
            break
        m = _PLAY_STEP_RE.match(str(step))
        if not m:
            continue
        card_name = m.group("card")
        hi = int(m.group("hi"))
        entry = _hand_entry(board, hi)
        if entry is not None:
            card_name = str(entry.get("name") or card_name)
            energy += _parse_energy_cost(entry.get("cost"))
        else:
            energy += 1
        block_gain += _est_block_from_card_name(card_name)
        damage += _est_damage_from_card_name(card_name)
    return block_gain, damage, energy


def score_turn_plan_candidate(
    plan: TurnPlanCandidate, board: dict[str, Any]
) -> tuple[int, int, int, tuple[str, ...]]:
    """Deterministic heuristic: block lethal, survive, cost-efficiency; tie = steps."""
    self_row = board.get("self") or {}
    hp = int(self_row.get("hp") or 0)
    block = int(self_row.get("block") or 0)
    energy = int(self_row.get("energy") or 0)
    incoming = incoming_attack_damage_from_board(board)
    plan_block, plan_damage, energy_spent = _analyze_plan_steps(plan.steps, board)
    effective_block = block + plan_block

    if incoming > 0:
        if effective_block >= incoming:
            block_score = 1000
        elif incoming >= hp:
            block_score = int(300 * effective_block / max(incoming, 1))
        else:
            block_score = int(600 * effective_block / max(incoming, 1))
    else:
        block_score = 200

    post_hit = max(0, incoming - effective_block)
    survive_score = max(0, min(hp, hp - post_hit) + (10 if post_hit == 0 else 0))

    energy_budget = max(1, energy)
    if energy_spent > energy_budget:
        eff_score = max(0, 50 - (energy_spent - energy_budget) * 20)
    else:
        spare = energy_budget - energy_spent
        eff_score = 80 + plan_damage * 5 + spare * 3 - energy_spent * 2

    return block_score, survive_score, int(eff_score), plan.steps


def composite_heuristic_score(components: tuple[int, int, int]) -> float:
    block_score, survive_score, eff_score = components
    return float(block_score * 1_000_000 + survive_score * 1_000 + eff_score)


def heuristic_score_distribution(scores: Sequence[float]) -> dict[str, Any]:
    n = len(scores)
    if not n:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "buckets": {"lt_1e5": 0, "1e5_1e6": 0, "1e6_11e6": 0, "ge_11e6": 0},
        }
    arr = np.sort(np.asarray(scores, dtype=float))
    buckets = {"lt_1e5": 0, "1e5_1e6": 0, "1e6_11e6": 0, "ge_11e6": 0}
    for x in arr:
        if x < 1e5:
            buckets["lt_1e5"] += 1
        elif x < 1e6:
            buckets["1e5_1e6"] += 1
        elif x < 1.1e6:
            buckets["1e6_11e6"] += 1
        else:
            buckets["ge_11e6"] += 1
    return {
        "n": n,
        "min": round(float(arr[0]), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr[-1]), 2),
        "buckets": buckets,
    }


def resolve_turn_plan_choice_cap(
    override: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Product cap for ``plan_id`` Choice (default 32, clamped to TypeSafe max 255)."""
    if override is not None:
        cap = int(override)
    else:
        env = os.environ if environ is None else environ
        raw = str(env.get(TURN_PLAN_CHOICE_CAP_ENV) or "").strip()
        cap = int(raw) if raw else DEFAULT_TURN_PLAN_CHOICE_CANDIDATES
    if cap < 1:
        raise ValueError(
            f"turn-plan Choice cap must be >= 1 (got {cap}); "
            f"platform max is {TYPESAFE_CHOICE_PLATFORM_MAX}"
        )
    return min(cap, TYPESAFE_CHOICE_PLATFORM_MAX)


def cap_plans_for_turn_plan_choice(
    plans: Sequence[TurnPlanCandidate],
    board: dict[str, Any] | None = None,
    *,
    max_choices: int | None = None,
) -> tuple[tuple[TurnPlanCandidate, ...], int, dict[str, Any], tuple[float, ...]]:
    """Heuristic top-K: block/survive/efficiency, tie-break on ``steps`` (not ``plan_id``)."""
    ctx = board if board is not None else _minimal_board_for_plan_scoring()
    limit = resolve_turn_plan_choice_cap(max_choices)
    if not plans:
        empty = heuristic_score_distribution(())
        return (), 0, empty, ()

    scored: list[tuple[tuple[int, int, int, tuple[str, ...]], TurnPlanCandidate]] = []
    composites: list[float] = []
    for plan in plans:
        components = score_turn_plan_candidate(plan, ctx)
        scored.append((components, plan))
        composites.append(
            composite_heuristic_score(components[:3])
        )
    scored.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            item[0][3],
        )
    )
    ordered = tuple(p for _c, p in scored)
    summary = heuristic_score_distribution(composites)
    comp_tuple = tuple(composites)
    if len(ordered) <= limit:
        return ordered, 0, summary, comp_tuple
    pruned = len(ordered) - limit
    return ordered[:limit], pruned, summary, comp_tuple


def build_combat_turn_plan_choice_question(
    plans: Sequence[TurnPlanCandidate],
    *,
    board: dict[str, Any] | None = None,
    instructions: str = COMBAT_TURN_PLAN_INSTRUCTIONS,
) -> dict[str, Any]:
    """Jev Choice payload: criteria keys are ``plan_id`` strings only."""
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {
            p.plan_id: _plan_criteria_summary(p, board=board) for p in plans
        },
    }


def jev_turn_plan_questions(
    board: dict[str, Any],
    plans: Sequence[TurnPlanCandidate],
    *,
    prompt_config: "TurnPlanPromptConfig | None" = None,
    bh_assist: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Question map for system_one with board-aware instructions and criteria."""
    from sts2_env.eval.bh_assist import bh_assist_instruction_suffix

    cfg = prompt_config or DEFAULT_TURN_PLAN_PROMPT_CONFIG
    instructions = _choice_instructions_with_board(board, COMBAT_TURN_PLAN_INSTRUCTIONS)
    if cfg.system_rules:
        instructions = COMBAT_TURN_PLAN_SYSTEM_RULES + " " + instructions
    if cfg.human_exemplars:
        instructions += " (See bundled exemplars in state when enabled.)"
    if bh_assist is not None:
        instructions += bh_assist_instruction_suffix(bh_assist)
    return {
        CHOICE_COMBAT_TURN_PLAN: build_combat_turn_plan_choice_question(
            plans, board=board, instructions=instructions
        )
    }


@dataclass
class TurnPlanPromptConfig:
    """Layered prompt toggles for turn-plan Choice."""

    system_rules: bool = True
    board_json: bool = True
    human_exemplars: bool = False


DEFAULT_TURN_PLAN_PROMPT_CONFIG = TurnPlanPromptConfig()


@dataclass
class TurnPlanRuntime:
    player_turn_id: int = -1
    replans: int = 0
    plan: TurnPlanCandidate | None = None
    step_index: int = 0
    intent_snapshot: dict[int, str] | None = None
    last_shadow: dict[str, Any] = field(default_factory=dict)
    telemetry_logged_turn: int = -1

    def reset_player_turn(self, turn_id: int) -> None:
        self.player_turn_id = int(turn_id)
        self.replans = 0
        self.clear_plan()

    def clear_plan(self) -> None:
        self.plan = None
        self.step_index = 0
        self.intent_snapshot = None


def runtime_for_env(env: Any) -> TurnPlanRuntime:
    key = "_combat_turn_plan_runtime"
    rt = getattr(env, key, None)
    if rt is None:
        rt = TurnPlanRuntime()
        setattr(env, key, rt)
    return rt


def player_turn_id(combat: CombatState) -> int:
    return int(getattr(combat, "turn_count", 0) or 0)


def living_enemy_intent_snapshot(combat: CombatState) -> dict[int, str]:
    return {
        int(enemy.combat_id): _intent_line(combat, enemy)
        for enemy in combat.enemies
        if enemy.is_alive
    }


def build_turn_plan_jev_state(
    board: dict[str, Any],
    *,
    prompt_config: TurnPlanPromptConfig | None = None,
    bh_assist: Any | None = None,
) -> dict[str, Any]:
    cfg = prompt_config or DEFAULT_TURN_PLAN_PROMPT_CONFIG
    state: dict[str, Any] = {
        "mode": "combat_turn_plan",
        "combat_snapshot": summarize_board_for_turn_plan_choice(board),
    }
    if cfg.board_json:
        state["board"] = board
    if cfg.human_exemplars:
        state["human_exemplars"] = []
    if bh_assist is not None:
        state["bh_assist"] = bh_assist.as_dict()
    return state


def _classify_adapter_error(exc: BaseException) -> str:
    from sts2_env.eval.combat_jev import classify_jev_error

    return classify_jev_error(exc)


def _fail_open_bh_v1(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[int, str | None]:
    del rng
    from sts2_env.eval.combat_jev import fail_open_bh_v1_required

    local, miss = fail_open_bh_v1_required(combat_model, combat_obs, combat_mask)
    return int(local), miss


def catastrophe_reason_for_telemetry(reason: str) -> str:
    """Map turn-plan catastrophe codes to ``CombatJevTelemetry`` reason keys."""
    from sts2_env.eval.combat_jev import (
        FAILOPEN_EMPTY_LIST,
        FAILOPEN_ERROR,
        FAILOPEN_ILLEGAL_PLAN,
        FAILOPEN_MISSING_HUNG_PPO,
        FAILOPEN_REPLAN_CAP,
        FAILOPEN_TIMEOUT,
    )

    mapping = {
        CATA_FAILOPEN_TIMEOUT: FAILOPEN_TIMEOUT,
        CATA_FAILOPEN_ERROR: FAILOPEN_ERROR,
        CATA_FAILOPEN_EMPTY: FAILOPEN_EMPTY_LIST,
        CATA_FAILOPEN_ILLEGAL_PLAN: FAILOPEN_ILLEGAL_PLAN,
        CATA_FAILOPEN_CAP: FAILOPEN_REPLAN_CAP,
        CATA_FAILOPEN_MISSING_HUNG_PPO: FAILOPEN_MISSING_HUNG_PPO,
    }
    return mapping.get(reason, FAILOPEN_ERROR)


def _log_turn_plan_telemetry(
    session: TurnPlanRuntime,
    telemetry: Any,
    *,
    fulfilled: bool,
    catastrophe_reason: str | None = None,
    error_detail: str | None = None,
) -> None:
    if session.telemetry_logged_turn == session.player_turn_id:
        return
    record_jev_turn_plan_turn(
        telemetry,
        fulfilled=fulfilled,
        catastrophe_reason=catastrophe_reason,
        error_detail=error_detail,
    )
    session.telemetry_logged_turn = session.player_turn_id


def _patch_replay_turn_trajectory(
    env: Any | None,
    combat: CombatState,
    mask: np.ndarray,
    session: TurnPlanRuntime,
) -> None:
    from sts2_env.eval.hold_turn_replay import replay_recorder_from_env

    rec = replay_recorder_from_env(env) if env is not None else None
    if rec is None:
        return
    board = serialize_combat_board_full(
        combat, mask, owner=combat.primary_player
    )
    rec.patch_last_turn_trajectory(board, replan_count=int(session.replans))


def _patch_replay_replan_count(env: Any | None, session: TurnPlanRuntime) -> None:
    from sts2_env.eval.hold_turn_replay import replay_recorder_from_env

    rec = replay_recorder_from_env(env) if env is not None else None
    if rec is None:
        return
    rec.patch_last_turn_replan_count(int(session.replans))


def record_turn_plan_replan_trigger(telemetry: Any, reason: str) -> None:
    if telemetry is None:
        return
    record = getattr(telemetry, "record_turn_plan_replan_trigger", None)
    if callable(record):
        record(str(reason))


def _increment_turn_plan_replan(
    session: TurnPlanRuntime,
    env: Any | None,
    telemetry: Any,
    reason: str,
) -> None:
    session.replans += 1
    record_turn_plan_replan_trigger(telemetry, reason)
    _patch_replay_replan_count(env, session)


def _catastrophe_fail_open(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
    reason: str,
    *,
    plan_id: str | None = None,
    error_detail: str | None = None,
) -> tuple[int, dict[str, Any], str, str | None]:
    from sts2_env.eval.combat_jev import FAILOPEN_ERROR

    local, bh_miss = _fail_open_bh_v1(combat_model, combat_obs, combat_mask, rng)
    effective_reason = (
        CATA_FAILOPEN_MISSING_HUNG_PPO if bh_miss else reason
    )
    tel_reason = catastrophe_reason_for_telemetry(effective_reason)
    tel_detail = None if bh_miss else error_detail
    if (
        not bh_miss
        and tel_reason == FAILOPEN_ERROR
        and error_detail
    ):
        tel_detail = error_detail
    shadow = {
        "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
        "turn_plan_catastrophe": True,
        "turn_plan_failopen_reason": effective_reason,
        "turn_plan_failopen": True,
        "executed_id": semantic_key_for_gym_action_from_obs(local, combat_mask),
    }
    if error_detail and not bh_miss:
        shadow["turn_plan_error_sample"] = error_detail[:320]
    if plan_id is not None:
        shadow["turn_plan_id"] = plan_id
    return local, shadow, tel_reason, tel_detail


def semantic_key_for_gym_action_from_obs(action: int, mask: np.ndarray) -> str:
    del mask
    return f"gym_a{int(action)}"


def _pick_plan_via_jev(
    adapter: Any,
    board: dict[str, Any],
    plans: Sequence[TurnPlanCandidate],
    *,
    prompt_config: TurnPlanPromptConfig | None = None,
    bh_assist: Any | None = None,
) -> tuple[TurnPlanCandidate | None, str | None, str | None]:
    from sts2_env.eval.combat_jev import format_turn_plan_error_sample

    if not plans:
        return None, CATA_FAILOPEN_EMPTY, None
    by_id = {p.plan_id: p for p in plans}
    cfg = prompt_config or DEFAULT_TURN_PLAN_PROMPT_CONFIG
    state = build_turn_plan_jev_state(
        board, prompt_config=cfg, bh_assist=bh_assist
    )
    questions = jev_turn_plan_questions(
        board, plans, prompt_config=cfg, bh_assist=bh_assist
    )
    try:
        answers = adapter.system_one(state, questions)
    except JevError as e:
        err = _classify_adapter_error(e)
        detail = format_turn_plan_error_sample(e) if err == CATA_FAILOPEN_ERROR else None
        return None, err, detail
    except Exception as e:
        err = _classify_adapter_error(e)
        detail = format_turn_plan_error_sample(e) if err == CATA_FAILOPEN_ERROR else None
        return None, err, detail
    raw = answers.get(CHOICE_COMBAT_TURN_PLAN) if isinstance(answers, dict) else None
    if raw is None:
        return (
            None,
            CATA_FAILOPEN_ERROR,
            f"missing {CHOICE_COMBAT_TURN_PLAN} in system_one answers",
        )
    if not isinstance(raw, JevAnswer):
        raw = JevAnswer(status="ok", choice=str(getattr(raw, "choice", raw)))
    plan_id = str(raw.choice or "").strip()
    if plan_id not in by_id:
        return None, CATA_FAILOPEN_ILLEGAL_PLAN, None
    return by_id[plan_id], None, None


def choose_combat_turn_plan_action(
    combat: CombatState,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
    combat_model: Any,
    *,
    adapter: Any,
    combat_obs: np.ndarray,
    env: Any | None = None,
    owner: Creature | None = None,
    prompt_config: TurnPlanPromptConfig | None = None,
    runtime: TurnPlanRuntime | None = None,
    telemetry: Any = None,
    turn_plan_choice_cap: int | None = None,
    bh_assist_config: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute turn-plan path: pick ``plan_id``, run steps, replan on triggers.

    Catastrophe fail-open (bh_v1) only for timeout/error/empty/illegal_plan/cap.
    Low confidence does **not** fail-open. ``pending_choice`` aborts the plan and
    uses bh_v1 fail-open (not turn_plan Choice).
    """
    mask = np.asarray(combat_mask)
    if runtime is not None:
        session = runtime
    elif env is not None:
        session = runtime_for_env(env)
    else:
        session = TurnPlanRuntime()
    cfg = prompt_config or DEFAULT_TURN_PLAN_PROMPT_CONFIG
    owner_creature = owner or combat.primary_player

    turn_id = player_turn_id(combat)
    if session.player_turn_id != turn_id:
        session.reset_player_turn(turn_id)

    if combat.pending_choice is not None:
        session.clear_plan()
        local, _bh_miss = _fail_open_bh_v1(combat_model, combat_obs, mask, rng)
        shadow = {
            "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
            "turn_plan_aborted": "pending_choice",
            "turn_plan_failopen": True,
            "executed_id": semantic_key_for_gym_action_from_obs(local, mask),
        }
        _log_turn_plan_telemetry(session, telemetry, fulfilled=False)
        session.last_shadow = shadow
        return local, shadow

    loops = 0
    while loops < 32:
        loops += 1
        if session.replans > MAX_REPLANS_PER_PLAYER_TURN:
            from sts2_env.eval.hold_turn_replay import replay_recorder_from_env

            session.clear_plan()
            replay_rec = replay_recorder_from_env(env) if env is not None else None
            cap_board = serialize_combat_board_full(
                combat, mask, owner=owner_creature
            )
            local, shadow, tel_reason, tel_detail = _catastrophe_fail_open(
                combat_model, combat_obs, mask, rng, CATA_FAILOPEN_CAP
            )
            if replay_rec is not None:
                replay_rec.record_replan_cap_catastrophe(
                    player_turn=turn_id,
                    board=cap_board,
                    replan_count=int(session.replans),
                    shadow=shadow,
                    fail_open_reason=str(
                        shadow.get("turn_plan_failopen_reason") or CATA_FAILOPEN_CAP
                    ),
                )
            _log_turn_plan_telemetry(
                session,
                telemetry,
                fulfilled=False,
                catastrophe_reason=tel_reason,
                error_detail=tel_detail,
            )
            session.last_shadow = shadow
            return local, shadow

        if session.plan is not None and session.step_index < len(session.plan.steps):
            key = session.plan.steps[session.step_index]
            legal = set(legal_semantic_keys(combat, mask, owner=owner_creature))
            resolved = remap_plan_step_semantic_key(key, legal)
            if resolved is None:
                _increment_turn_plan_replan(
                    session, env, telemetry, "illegal_step"
                )
                session.clear_plan()
                continue
            key = resolved
            if living_enemy_intent_snapshot(combat) != (session.intent_snapshot or {}):
                _increment_turn_plan_replan(
                    session, env, telemetry, "intent_drift"
                )
                session.clear_plan()
                continue
            action = gym_action_for_semantic_key(
                combat, mask, key, owner=owner_creature
            )
            if action is None:
                _increment_turn_plan_replan(
                    session, env, telemetry, "action_map_none"
                )
                session.clear_plan()
                continue
            session.step_index += 1
            plan_completed = key == SEMANTIC_END_TURN or session.step_index >= len(
                session.plan.steps
            )
            if plan_completed:
                session.clear_plan()
            shadow = {
                "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
                "turn_plan_step": key,
                "turn_plan_failopen": False,
                "executed_id": key,
            }
            if plan_completed:
                _log_turn_plan_telemetry(session, telemetry, fulfilled=True)
                _patch_replay_turn_trajectory(env, combat, mask, session)
            session.last_shadow = shadow
            return int(action), shadow

        legal_keys = legal_semantic_keys(combat, mask, owner=owner_creature)
        board = serialize_combat_board_full(combat, mask, owner=owner_creature)
        raw_plans = enumerate_candidate_plans(legal_keys)
        plans, pruned_n, score_summary, score_composites = cap_plans_for_turn_plan_choice(
            raw_plans, board, max_choices=turn_plan_choice_cap
        )
        if pruned_n:
            record_turn_plan_choice_prune(telemetry, pruned_n)
        record_turn_plan_heuristic_scores(telemetry, score_summary, score_composites)
        from sts2_env.eval.bh_assist import try_turn_plan_bh_assist

        assist = try_turn_plan_bh_assist(
            board,
            legal_keys,
            config=bh_assist_config,
            combat_obs=combat_obs,
            combat=combat,
            mask=mask,
            owner=owner_creature,
        )
        picked, err, err_detail = _pick_plan_via_jev(
            adapter,
            board,
            plans,
            prompt_config=cfg,
            bh_assist=assist,
        )
        from sts2_env.eval.hold_turn_replay import replay_recorder_from_env

        replay_rec = replay_recorder_from_env(env) if env is not None else None
        if err is not None:
            session.clear_plan()
            local, shadow, tel_reason, tel_detail = _catastrophe_fail_open(
                combat_model,
                combat_obs,
                mask,
                rng,
                err,
                error_detail=err_detail,
            )
            if replay_rec is not None:
                replay_rec.record_plan_choice(
                    player_turn=turn_id,
                    plans=plans,
                    board=board,
                    pruned_plan_count=pruned_n,
                    picked_plan_id=None,
                    pick_error=err,
                    bh_assist=assist,
                    shadow=shadow,
                    replan_count=int(session.replans),
                    replan_cap_hit=err == CATA_FAILOPEN_CAP,
                    fail_open=True,
                    fail_open_reason=str(
                        shadow.get("turn_plan_failopen_reason") or err
                    ),
                )
                _patch_replay_turn_trajectory(env, combat, mask, session)
            _log_turn_plan_telemetry(
                session,
                telemetry,
                fulfilled=False,
                catastrophe_reason=tel_reason,
                error_detail=tel_detail,
            )
            session.last_shadow = shadow
            return local, shadow
        if replay_rec is not None and picked is not None:
            replay_rec.record_plan_choice(
                player_turn=turn_id,
                plans=plans,
                board=board,
                pruned_plan_count=pruned_n,
                picked_plan_id=picked.plan_id,
                pick_error=None,
                bh_assist=assist,
                shadow=None,
                replan_count=int(session.replans),
                replan_cap_hit=False,
                fail_open=False,
                fail_open_reason=None,
            )
        assert picked is not None
        session.plan = picked
        session.intent_snapshot = living_enemy_intent_snapshot(combat)
        session.step_index = 0

    session.clear_plan()
    local, shadow, tel_reason, tel_detail = _catastrophe_fail_open(
        combat_model,
        combat_obs,
        mask,
        rng,
        CATA_FAILOPEN_ERROR,
        error_detail="turn-plan loop cap exceeded without plan pick",
    )
    _log_turn_plan_telemetry(
        session,
        telemetry,
        fulfilled=False,
        catastrophe_reason=tel_reason,
        error_detail=tel_detail,
    )
    session.last_shadow = shadow
    return local, shadow


def record_turn_plan_heuristic_scores(
    telemetry: Any,
    summary: dict[str, Any],
    composites: Sequence[float],
) -> None:
    if telemetry is None or int(summary.get("n") or 0) <= 0:
        return
    record = getattr(telemetry, "record_turn_plan_heuristic_scores", None)
    if callable(record):
        record(summary, composites)


def record_turn_plan_choice_prune(telemetry: Any, pruned_count: int) -> None:
    if telemetry is None or int(pruned_count) <= 0:
        return
    record = getattr(telemetry, "record_turn_plan_choice_prune", None)
    if callable(record):
        record(int(pruned_count))


def record_jev_turn_plan_turn(
    telemetry: Any,
    *,
    fulfilled: bool,
    catastrophe_reason: str | None = None,
    error_detail: str | None = None,
) -> None:
    """Telemetry hook for ``--combat-policy jev-turn`` (tip② loop calls each turn).

    Pass ``catastrophe_reason`` only for timeout/error/empty_list/illegal_plan/
    replan_cap fail-open. ``low_conf`` guardrail: ``fulfilled=False``,
    ``catastrophe_reason=None``.
    """
    if telemetry is None:
        return
    record = getattr(telemetry, "record_turn_plan_turn", None)
    if callable(record):
        record(
            fulfilled=fulfilled,
            catastrophe_reason=catastrophe_reason,
            error_detail=error_detail,
        )


__all__ = [
    "CATA_FAILOPEN_CAP",
    "CATA_FAILOPEN_EMPTY",
    "CATA_FAILOPEN_ERROR",
    "CATA_FAILOPEN_MISSING_HUNG_PPO",
    "CATA_FAILOPEN_ILLEGAL_PLAN",
    "CATA_FAILOPEN_TIMEOUT",
    "CHOICE_COMBAT_TURN_PLAN",
    "COMBAT_TURN_PLAN_INSTRUCTIONS",
    "COMBAT_TURN_PLAN_SYSTEM_RULES",
    "DEFAULT_TURN_PLAN_PROMPT_CONFIG",
    "DEFAULT_TURN_PLAN_CHOICE_CANDIDATES",
    "MAX_CANDIDATE_PLANS",
    "MAX_PLAN_STEPS",
    "MAX_TURN_PLAN_CHOICE_CANDIDATES",
    "TURN_PLAN_CHOICE_CAP_ENV",
    "TYPESAFE_CHOICE_PLATFORM_MAX",
    "MAX_REPLANS_PER_PLAYER_TURN",
    "SEMANTIC_END_TURN",
    "TurnPlanCandidate",
    "TurnPlanPromptConfig",
    "TurnPlanRuntime",
    "build_combat_turn_plan_choice_question",
    "cap_plans_for_turn_plan_choice",
    "composite_heuristic_score",
    "heuristic_score_distribution",
    "incoming_attack_damage_from_board",
    "resolve_turn_plan_choice_cap",
    "score_turn_plan_candidate",
    "build_turn_plan_jev_state",
    "summarize_board_for_turn_plan_choice",
    "catastrophe_reason_for_telemetry",
    "choose_combat_turn_plan_action",
    "enumerate_candidate_plans",
    "gym_action_for_semantic_key",
    "jev_turn_plan_questions",
    "record_jev_turn_plan_turn",
    "record_turn_plan_choice_prune",
    "record_turn_plan_heuristic_scores",
    "record_turn_plan_replan_trigger",
    "plan_step_action_signature",
    "remap_plan_step_semantic_key",
    "legal_semantic_keys",
    "living_enemy_intent_snapshot",
    "runtime_for_env",
    "semantic_key_for_gym_action",
    "serialize_combat_board_full",
]
