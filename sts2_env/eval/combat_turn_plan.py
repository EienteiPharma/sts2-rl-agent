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

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

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
MAX_REPLANS_PER_PLAYER_TURN = 3

CATA_FAILOPEN_TIMEOUT = "timeout"
CATA_FAILOPEN_ERROR = "error"
CATA_FAILOPEN_EMPTY = "empty"
CATA_FAILOPEN_ILLEGAL_PLAN = "illegal_plan"
CATA_FAILOPEN_CAP = "cap_exceeded"

COMBAT_TURN_PLAN_SYSTEM_RULES = (
    "Survival first: block or prevent telegraphed enemy intent damage before "
    "greedy damage. Prefer enumerated plan_ids only."
)

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
    *,
    prompt_config: "TurnPlanPromptConfig | None" = None,
) -> dict[str, dict[str, Any]]:
    """Question map for system_one (board may be omitted when layers are off)."""
    del board
    cfg = prompt_config or TurnPlanPromptConfig()
    instructions = COMBAT_TURN_PLAN_INSTRUCTIONS
    if cfg.system_rules:
        instructions = COMBAT_TURN_PLAN_SYSTEM_RULES + " " + instructions
    if cfg.human_exemplars:
        instructions += " (See bundled exemplars in state when enabled.)"
    return {
        CHOICE_COMBAT_TURN_PLAN: build_combat_turn_plan_choice_question(
            plans, instructions=instructions
        )
    }


@dataclass
class TurnPlanPromptConfig:
    """Layered prompt toggles (defaults OFF for first smoke)."""

    system_rules: bool = False
    board_json: bool = False
    human_exemplars: bool = False


@dataclass
class TurnPlanRuntime:
    player_turn_id: int = -1
    replans: int = 0
    plan: TurnPlanCandidate | None = None
    step_index: int = 0
    intent_snapshot: dict[int, str] | None = None
    last_shadow: dict[str, Any] = field(default_factory=dict)

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
) -> dict[str, Any]:
    cfg = prompt_config or TurnPlanPromptConfig()
    state: dict[str, Any] = {"mode": "combat_turn_plan"}
    if cfg.board_json:
        state["board"] = board
    if cfg.human_exemplars:
        state["human_exemplars"] = []
    return state


def _classify_adapter_error(exc: BaseException) -> str:
    from sts2_env.eval.combat_jev import classify_jev_error

    return classify_jev_error(exc)


def _fail_open_bh_v1(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    from sts2_env.eval.combat_jev import fail_open_local

    return int(fail_open_local(combat_model, combat_obs, combat_mask, rng))


def _catastrophe_fail_open(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
    reason: str,
    *,
    plan_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    local = _fail_open_bh_v1(combat_model, combat_obs, combat_mask, rng)
    shadow = {
        "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
        "turn_plan_catastrophe": True,
        "turn_plan_failopen_reason": reason,
        "turn_plan_failopen": True,
        "executed_id": semantic_key_for_gym_action_from_obs(local, combat_mask),
    }
    if plan_id is not None:
        shadow["turn_plan_id"] = plan_id
    return local, shadow


def semantic_key_for_gym_action_from_obs(action: int, mask: np.ndarray) -> str:
    del mask
    return f"gym_a{int(action)}"


def _pick_plan_via_jev(
    adapter: Any,
    board: dict[str, Any],
    plans: Sequence[TurnPlanCandidate],
    *,
    prompt_config: TurnPlanPromptConfig | None = None,
) -> tuple[TurnPlanCandidate | None, str | None]:
    if not plans:
        return None, CATA_FAILOPEN_EMPTY
    by_id = {p.plan_id: p for p in plans}
    cfg = prompt_config or TurnPlanPromptConfig()
    state = build_turn_plan_jev_state(board, prompt_config=cfg)
    questions = jev_turn_plan_questions(board, plans, prompt_config=cfg)
    try:
        answers = adapter.system_one(state, questions)
    except JevError as e:
        return None, _classify_adapter_error(e)
    except Exception as e:
        return None, _classify_adapter_error(e)
    raw = answers.get(CHOICE_COMBAT_TURN_PLAN) if isinstance(answers, dict) else None
    if raw is None:
        return None, CATA_FAILOPEN_ERROR
    if not isinstance(raw, JevAnswer):
        raw = JevAnswer(status="ok", choice=str(getattr(raw, "choice", raw)))
    plan_id = str(raw.choice or "").strip()
    if plan_id not in by_id:
        return None, CATA_FAILOPEN_ILLEGAL_PLAN
    return by_id[plan_id], None


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
    cfg = prompt_config or TurnPlanPromptConfig()
    owner_creature = owner or combat.primary_player

    if combat.pending_choice is not None:
        session.clear_plan()
        local = _fail_open_bh_v1(combat_model, combat_obs, mask, rng)
        shadow = {
            "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
            "turn_plan_aborted": "pending_choice",
            "turn_plan_failopen": True,
            "executed_id": semantic_key_for_gym_action_from_obs(local, mask),
        }
        session.last_shadow = shadow
        return local, shadow

    turn_id = player_turn_id(combat)
    if session.player_turn_id != turn_id:
        session.reset_player_turn(turn_id)

    loops = 0
    while loops < 32:
        loops += 1
        if session.replans > MAX_REPLANS_PER_PLAYER_TURN:
            session.clear_plan()
            local, shadow = _catastrophe_fail_open(
                combat_model, combat_obs, mask, rng, CATA_FAILOPEN_CAP
            )
            session.last_shadow = shadow
            return local, shadow

        if session.plan is not None and session.step_index < len(session.plan.steps):
            key = session.plan.steps[session.step_index]
            legal = set(legal_semantic_keys(combat, mask, owner=owner_creature))
            if key not in legal:
                session.replans += 1
                session.clear_plan()
                continue
            if living_enemy_intent_snapshot(combat) != (session.intent_snapshot or {}):
                session.replans += 1
                session.clear_plan()
                continue
            action = gym_action_for_semantic_key(
                combat, mask, key, owner=owner_creature
            )
            if action is None:
                session.replans += 1
                session.clear_plan()
                continue
            session.step_index += 1
            if key == SEMANTIC_END_TURN or session.step_index >= len(session.plan.steps):
                session.clear_plan()
            shadow = {
                "shadow_decision": CHOICE_COMBAT_TURN_PLAN,
                "turn_plan_step": key,
                "turn_plan_failopen": False,
                "executed_id": key,
            }
            session.last_shadow = shadow
            return int(action), shadow

        legal_keys = legal_semantic_keys(combat, mask, owner=owner_creature)
        plans = enumerate_candidate_plans(legal_keys)
        board = serialize_combat_board_full(combat, mask, owner=owner_creature)
        picked, err = _pick_plan_via_jev(adapter, board, plans, prompt_config=cfg)
        if err is not None:
            session.clear_plan()
            local, shadow = _catastrophe_fail_open(
                combat_model, combat_obs, mask, rng, err
            )
            session.last_shadow = shadow
            return local, shadow
        assert picked is not None
        session.plan = picked
        session.intent_snapshot = living_enemy_intent_snapshot(combat)
        session.step_index = 0

    session.clear_plan()
    local, shadow = _catastrophe_fail_open(
        combat_model, combat_obs, mask, rng, CATA_FAILOPEN_ERROR
    )
    session.last_shadow = shadow
    return local, shadow


__all__ = [
    "CATA_FAILOPEN_CAP",
    "CATA_FAILOPEN_EMPTY",
    "CATA_FAILOPEN_ERROR",
    "CATA_FAILOPEN_ILLEGAL_PLAN",
    "CATA_FAILOPEN_TIMEOUT",
    "CHOICE_COMBAT_TURN_PLAN",
    "COMBAT_TURN_PLAN_INSTRUCTIONS",
    "COMBAT_TURN_PLAN_SYSTEM_RULES",
    "MAX_CANDIDATE_PLANS",
    "MAX_PLAN_STEPS",
    "MAX_REPLANS_PER_PLAYER_TURN",
    "SEMANTIC_END_TURN",
    "TurnPlanCandidate",
    "TurnPlanPromptConfig",
    "TurnPlanRuntime",
    "build_combat_turn_plan_choice_question",
    "build_turn_plan_jev_state",
    "choose_combat_turn_plan_action",
    "enumerate_candidate_plans",
    "gym_action_for_semantic_key",
    "jev_turn_plan_questions",
    "legal_semantic_keys",
    "living_enemy_intent_snapshot",
    "runtime_for_env",
    "semantic_key_for_gym_action",
    "serialize_combat_board_full",
]
