"""Experimental combat-step Jev Choice bypass (failed HOLD; not hang).

Lab lock 2026-09-23: HOLD smoke on tip ``2b1dcf4`` collapsed
(A ppo 74.2 / 98.9 / Boss 49.4 vs B jev 18.6 / 36.7 / Boss 0.6;
failopen_rate 41.2% mostly ``low_conf``). Damage was from Jev-chosen
steps, not insufficient fail-open. Archive: ``docs/COMBAT_JEV_HOLD_FAIL.md``.

Locked contract (unchanged; flag kept, default remains ``--combat-policy ppo``):
- Choice name: ``combat_step_choice``
- Options: code-enumerated legal play-card / use-potion / end-turn only
- Confidence default **0.35**; missing or below → fail-open
- Fail-open order: hung PPO (bh_v1) → random legal
- Reasons: ``timeout`` | ``error`` | ``bad_id`` | ``low_conf`` | ``empty_list``

Default eval path remains ``--combat-policy ppo``. This module is unused
until the operator opts into experimental ``--combat-policy jev``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Cards must finish registering before CombatState / action_space / observation
# import core.combat; otherwise core.combat ↔ cards is a circular ImportError.
import sts2_env.cards  # noqa: F401

from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PowerId
from sts2_env.eval.jev_fallback import apply_choice_confidence
from sts2_env.eval.jev_types import JevAnswer, JevError
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    is_potion_action,
)
from sts2_env.gym_env.observation import encode_observation

CHOICE_COMBAT_STEP = "combat_step_choice"
COMBAT_JEV_CONF_MIN = 0.45
COMBAT_JEV_HAND_TRUNCATE = 8
COMBAT_JEV_MONSTER_TRUNCATE = 3
COMBAT_JEV_INSTRUCTIONS = (
    "You may only choose a key that appears on the legal shortlist. "
    "Priority: block or prevent this-turn intent damage and survive first; "
    "efficiency second. "
    "If the target is unclear or two options look equally good, prefer low "
    "confidence and let fail-open handle it; do not hard-pick."
)

FAILOPEN_TIMEOUT = "timeout"
FAILOPEN_ERROR = "error"
FAILOPEN_BAD_ID = "bad_id"
FAILOPEN_LOW_CONF = "low_conf"
FAILOPEN_EMPTY_LIST = "empty_list"
FAILOPEN_ILLEGAL_PLAN = "illegal_plan"
FAILOPEN_REPLAN_CAP = "replan_cap"
FAILOPEN_MISSING_HUNG_PPO = "missing_hung_ppo"
TURN_PLAN_ERROR_SAMPLE_MAX = 8
COMBAT_JEV_FAILOPEN_REASONS = (
    FAILOPEN_TIMEOUT,
    FAILOPEN_ERROR,
    FAILOPEN_BAD_ID,
    FAILOPEN_LOW_CONF,
    FAILOPEN_EMPTY_LIST,
)
# ``--combat-policy jev-turn`` catastrophe fail-open only (not ``low_conf`` guardrail).
COMBAT_JEV_TURN_CATASTROPHE_REASONS = (
    FAILOPEN_TIMEOUT,
    FAILOPEN_ERROR,
    FAILOPEN_EMPTY_LIST,
    FAILOPEN_ILLEGAL_PLAN,
    FAILOPEN_REPLAN_CAP,
    FAILOPEN_MISSING_HUNG_PPO,
)

_KEY_POWERS = (
    (PowerId.STRENGTH, "str"),
    (PowerId.DEXTERITY, "dex"),
    (PowerId.VULNERABLE, "vuln"),
    (PowerId.WEAK, "weak"),
    (PowerId.FRAIL, "frail"),
    (PowerId.ARTIFACT, "artifact"),
)


def combat_action_id(action: int) -> str:
    """Stable shortlist id for a gym combat action index."""
    return f"a{int(action)}"


def parse_combat_action_id(action_id: str | None) -> int | None:
    if action_id is None:
        return None
    raw = str(action_id).strip()
    if raw.startswith("a") and raw[1:].isdigit():
        return int(raw[1:])
    if raw.isdigit():
        return int(raw)
    return None


def _label(obj: Any, *, attr: str = "name") -> str:
    if obj is None:
        return "?"
    val = getattr(obj, attr, obj)
    return str(getattr(val, "name", val))


def _cost_text(card: Any) -> str:
    if getattr(card, "has_energy_cost_x", False):
        return "X"
    return str(getattr(card, "cost", "?"))


def _powers_line(creature: Creature) -> str:
    parts: list[str] = []
    for pid, label in _KEY_POWERS:
        amt = int(creature.get_power_amount(pid) or 0)
        if amt:
            parts.append(f"{label}{amt}")
    return " ".join(parts)


def _hp_ratio_pct(creature: Creature) -> int:
    return int(round(100.0 * int(creature.current_hp) / max(int(creature.max_hp), 1)))


def _player_context_line(
    combat: CombatState,
    acting: Creature,
    *,
    end_turn_legal: bool,
) -> str:
    return (
        f"hp={_hp_ratio_pct(acting)}% "
        f"energy={int(combat.current_energy)} "
        f"end_turn={'yes' if end_turn_legal else 'no'}"
    )


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
        name = _label(it).lower()
        dmg = int(getattr(intent, "damage", 0) or 0)
        hits = int(getattr(intent, "hits", 1) or 1)
        if dmg > 0 and hits > 1:
            bits.append(f"{name} {dmg}x{hits}")
        elif dmg > 0:
            bits.append(f"{name} {dmg}")
        else:
            bits.append(name)
    return "/".join(bits) if bits else "unknown"


def _intent_for_enemies(combat: CombatState, enemies: list[Creature]) -> str:
    if not enemies:
        return "none"
    parts = [_intent_line(combat, enemy) for enemy in enemies if enemy.is_alive]
    return " | ".join(parts) if parts else "none"


def summarize_combat_action(
    combat: CombatState,
    action: int,
    *,
    owner: Creature | None = None,
    end_turn_legal: bool = True,
) -> str:
    """One-line shortlist summary for combat_step_choice criteria."""
    acting = owner or combat.primary_player
    owner_state = combat.combat_player_state_for(acting)
    hand = owner_state.hand if owner_state is not None else combat.hand
    potions = owner_state.potions if owner_state is not None else combat.potions
    alive = [e for e in combat.enemies if e.is_alive]
    ctx = _player_context_line(combat, acting, end_turn_legal=end_turn_legal)
    if int(action) == ACTION_END_TURN:
        intent = _intent_for_enemies(combat, alive)
        return f"end turn | intent: {intent} | {ctx}"
    if is_potion_action(int(action)):
        slot, tgt = action_to_potion_and_target(int(action))
        name = "?"
        if slot is not None and 0 <= slot < len(potions) and potions[slot] is not None:
            name = str(getattr(potions[slot], "potion_id", potions[slot]))
        if tgt is None:
            intent = _intent_for_enemies(combat, alive)
            return f"use potion {name} (self) | intent: {intent} | {ctx}"
        enemy = combat.enemies[tgt] if 0 <= tgt < len(combat.enemies) else None
        tgt_label = _label(enemy, attr="monster_id")
        intent = _intent_line(combat, enemy) if enemy is not None and enemy.is_alive else "none"
        return f"use potion {name} -> {tgt_label} | intent: {intent} | {ctx}"
    hand_i, tgt = action_to_card_and_target(int(action))
    if hand_i is None:
        return f"action {action} | {ctx}"
    card = hand[hand_i] if 0 <= hand_i < len(hand) else None
    name = _label(getattr(card, "card_id", None)) if card is not None else f"hand[{hand_i}]"
    cost = _cost_text(card) if card is not None else "?"
    if tgt is None:
        intent = _intent_for_enemies(combat, alive)
        return f"play {name} cost {cost} (self) | intent: {intent} | {ctx}"
    enemy = combat.enemies[tgt] if 0 <= tgt < len(combat.enemies) else None
    tgt_label = _label(enemy, attr="monster_id")
    intent = _intent_line(combat, enemy) if enemy is not None and enemy.is_alive else "none"
    return f"play {name} cost {cost} -> {tgt_label} | intent: {intent} | {ctx}"


def enumerate_legal_combat_actions(
    combat: CombatState,
    mask: np.ndarray,
    *,
    owner: Creature | None = None,
) -> list[tuple[str, int, str]]:
    """Legal play-card / use-potion / end-turn options only.

    Pending-choice masks are not combat-step options (caller should skip Jev).
    """
    if combat.pending_choice is not None:
        return []
    mask_arr = np.asarray(mask)
    end_turn_legal = (
        int(mask_arr[ACTION_END_TURN]) == 1 if mask_arr.size > ACTION_END_TURN else False
    )
    out: list[tuple[str, int, str]] = []
    for action in np.flatnonzero(mask_arr == 1):
        idx = int(action)
        if idx != ACTION_END_TURN and not is_potion_action(idx):
            hand_i, _tgt = action_to_card_and_target(idx)
            if hand_i is None:
                continue
        action_id = combat_action_id(idx)
        out.append(
            (
                action_id,
                idx,
                summarize_combat_action(
                    combat, idx, owner=owner, end_turn_legal=end_turn_legal
                ),
            )
        )
    return out


def compress_combat_state(
    combat: CombatState,
    *,
    owner: Creature | None = None,
    end_turn_legal: bool = True,
) -> dict[str, Any]:
    """Compressed Jev state: self / enemies / turn. Truncate hand>8, monsters>3."""
    acting = owner or combat.primary_player
    owner_state = combat.combat_player_state_for(acting)
    hand = list(owner_state.hand if owner_state is not None else combat.hand)
    truncated_hand = len(hand) > COMBAT_JEV_HAND_TRUNCATE
    hand = hand[:COMBAT_JEV_HAND_TRUNCATE]
    monsters = [e for e in combat.enemies if e.is_alive]
    truncated_monsters = len(monsters) > COMBAT_JEV_MONSTER_TRUNCATE
    monsters = monsters[:COMBAT_JEV_MONSTER_TRUNCATE]
    return {
        "self": {
            "hp": int(acting.current_hp),
            "max_hp": int(acting.max_hp),
            "energy": int(combat.current_energy),
            "hand": [
                {
                    "name": _label(getattr(card, "card_id", None)),
                    "cost": _cost_text(card),
                    "playable": bool(combat.can_play_card(card)),
                }
                for card in hand
            ],
            "buffs": _powers_line(acting),
        },
        "enemies": [
            {
                "name": _label(enemy, attr="monster_id"),
                "hp": int(enemy.current_hp),
                "max_hp": int(enemy.max_hp),
                "intent": _intent_line(combat, enemy),
                "vuln": int(enemy.get_power_amount(PowerId.VULNERABLE) or 0),
                "weak": int(enemy.get_power_amount(PowerId.WEAK) or 0),
            }
            for enemy in monsters
        ],
        "turn": {
            "player_turn_index": int(getattr(combat, "round_number", 1) or 1),
            "end_turn_legal": bool(end_turn_legal),
        },
        "truncated": {
            "hand": truncated_hand,
            "monsters": truncated_monsters,
        },
    }


def hung_ppo_local(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
) -> int | None:
    """Hung bh_v1 predict. None if the zip is unavailable."""
    if combat_model is None:
        return None
    local, _ = combat_model.predict(
        combat_obs, action_masks=combat_mask, deterministic=True
    )
    local = int(local)
    return max(0, min(local, ACTION_SPACE_SIZE - 1))


def _legal_end_turn_fallback(combat_mask: np.ndarray) -> int:
    mask = np.asarray(combat_mask)
    if mask.size > ACTION_END_TURN and int(mask[ACTION_END_TURN]) == 1:
        return ACTION_END_TURN
    valid = np.flatnonzero(mask == 1)
    if valid.size:
        return int(valid[0])
    return ACTION_END_TURN


def fail_open_bh_v1_required(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
) -> tuple[int, str | None]:
    """Turn-plan catastrophe fail-open: hung bh_v1 only (no random legal fallback)."""
    if combat_model is None:
        return _legal_end_turn_fallback(combat_mask), FAILOPEN_MISSING_HUNG_PPO
    try:
        local = hung_ppo_local(combat_model, combat_obs, combat_mask)
    except Exception:
        local = None
    if local is None:
        return _legal_end_turn_fallback(combat_mask), FAILOPEN_MISSING_HUNG_PPO
    return local, None


def format_turn_plan_error_sample(exc: BaseException, *, max_len: int = 320) -> str:
    """Bounded type+message (+ one traceback line) for jev-turn telemetry."""
    import traceback

    head = f"{type(exc).__name__}: {exc}"
    tb = traceback.format_exc().strip().splitlines()
    tail = ""
    if len(tb) >= 2:
        tail = f" | {tb[-1].strip()}"
    text = (head + tail).replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def fail_open_local(
    combat_model: Any,
    combat_obs: np.ndarray,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    """Fail-open: hung PPO, else random legal combat action."""
    local = None
    try:
        local = hung_ppo_local(combat_model, combat_obs, combat_mask)
    except Exception:
        local = None
    if local is not None:
        return local
    valid = np.flatnonzero(np.asarray(combat_mask) == 1)
    if valid.size == 0:
        return ACTION_END_TURN
    return int(rng.choice(valid))


def classify_jev_error(exc: BaseException) -> str:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timeout" in msg or "timed out" in msg or name == "timeouterror":
        return FAILOPEN_TIMEOUT
    return FAILOPEN_ERROR


@dataclass
class CombatJevTelemetry:
    """Sentry counters for the combat-Jev bypass (suite-level)."""

    calls: int = 0
    fail_open: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    failopen_reason: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in COMBAT_JEV_FAILOPEN_REASONS}
    )
    turn_plan_turns: int = 0
    turn_plan_fulfilled: int = 0
    turn_plan_catastrophe: int = 0
    catastrophe_failopen_reason: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS}
    )
    turn_plan_error_samples: list[str] = field(default_factory=list)
    turn_plan_pruned_count: int = 0
    turn_plan_heuristic_score_samples: list[float] = field(default_factory=list)
    turn_plan_replan_trigger: dict[str, int] = field(default_factory=dict)

    def mark(self) -> tuple[int, int]:
        return (self.calls, self.fail_open)

    def record(self, latency_ms: float, fail_reason: str | None = None) -> None:
        self.calls += 1
        self.latencies_ms.append(float(latency_ms))
        if fail_reason:
            self.fail_open += 1
            key = fail_reason if fail_reason in self.failopen_reason else FAILOPEN_ERROR
            self.failopen_reason[key] = int(self.failopen_reason.get(key, 0)) + 1

    def mark_turn_plan(self) -> tuple[int, int, int]:
        return (self.turn_plan_turns, self.turn_plan_fulfilled, self.turn_plan_catastrophe)

    def record_turn_plan_choice_prune(self, pruned_count: int) -> None:
        n = int(pruned_count)
        if n > 0:
            self.turn_plan_pruned_count += n

    def record_turn_plan_replan_trigger(self, reason: str) -> None:
        key = str(reason).strip() or "unknown"
        self.turn_plan_replan_trigger[key] = (
            int(self.turn_plan_replan_trigger.get(key, 0)) + 1
        )

    def record_turn_plan_heuristic_scores(
        self,
        summary: dict[str, Any],
        composites: Sequence[float],
    ) -> None:
        del summary
        if not composites:
            return
        self.turn_plan_heuristic_score_samples.extend(float(x) for x in composites)
        if len(self.turn_plan_heuristic_score_samples) > 4096:
            self.turn_plan_heuristic_score_samples = (
                self.turn_plan_heuristic_score_samples[-4096:]
            )

    def record_turn_plan_error_sample(self, detail: str) -> None:
        text = " ".join(str(detail).split())
        if not text:
            return
        if text in self.turn_plan_error_samples:
            return
        if len(self.turn_plan_error_samples) >= TURN_PLAN_ERROR_SAMPLE_MAX:
            return
        self.turn_plan_error_samples.append(text[:320])

    def record_turn_plan_turn(
        self,
        *,
        fulfilled: bool,
        catastrophe_reason: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        """One ``jev-turn`` player turn (``--combat-policy jev-turn``).

        ``catastrophe_reason`` must be in ``COMBAT_JEV_TURN_CATASTROPHE_REASONS``
        (timeout/error/empty_list/illegal_plan/replan_cap). ``low_conf`` guardrail
        is not catastrophe: pass ``fulfilled=False`` and ``catastrophe_reason=None``.
        """
        self.turn_plan_turns += 1
        if catastrophe_reason is not None:
            key = (
                catastrophe_reason
                if catastrophe_reason in self.catastrophe_failopen_reason
                else FAILOPEN_ERROR
            )
            self.turn_plan_catastrophe += 1
            self.catastrophe_failopen_reason[key] = int(
                self.catastrophe_failopen_reason.get(key, 0)
            ) + 1
            if key == FAILOPEN_ERROR and error_detail:
                self.record_turn_plan_error_sample(error_detail)
        elif fulfilled:
            self.turn_plan_fulfilled += 1

    def _turn_plan_report_rates(self) -> dict[str, Any]:
        turns = int(self.turn_plan_turns)
        fulfilled = int(self.turn_plan_fulfilled)
        catastrophe = int(self.turn_plan_catastrophe)
        return {
            "turn_plan_turns": turns,
            "turn_plan_fulfilled": fulfilled,
            "turn_plan_catastrophe_failopen": catastrophe,
            "jev_fulfilled_rate": round(fulfilled / turns, 4) if turns else 0.0,
            "catastrophe_failopen_rate": round(catastrophe / turns, 4) if turns else 0.0,
            "jev_turn_catastrophe_reason": {
                k: int(self.catastrophe_failopen_reason.get(k, 0))
                for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS
            },
            "jev_turn_catastrophe_error_samples": list(self.turn_plan_error_samples),
            "pruned_count": int(self.turn_plan_pruned_count),
            "turn_plan_heuristic_score": self._turn_plan_heuristic_score_report(),
            "turn_plan_replan_trigger": dict(self.turn_plan_replan_trigger),
        }

    def _turn_plan_heuristic_score_report(self) -> dict[str, Any]:
        from sts2_env.eval.combat_turn_plan import heuristic_score_distribution

        return heuristic_score_distribution(self.turn_plan_heuristic_score_samples)

    def episode_fields(self, start: tuple[int, int]) -> dict[str, int]:
        c0, f0 = start
        return {
            "combat_jev_calls": self.calls - c0,
            "combat_jev_fail_open": self.fail_open - f0,
        }

    def episode_turn_plan_fields(self, start: tuple[int, int, int]) -> dict[str, int]:
        t0, f0, c0 = start
        return {
            "turn_plan_turns": self.turn_plan_turns - t0,
            "turn_plan_fulfilled": self.turn_plan_fulfilled - f0,
            "turn_plan_catastrophe_failopen": self.turn_plan_catastrophe - c0,
        }

    def turn_plan_snapshot(self) -> tuple[int, int, int]:
        return (
            int(self.turn_plan_turns),
            int(self.turn_plan_fulfilled),
            int(self.turn_plan_catastrophe),
        )

    def turn_plan_reason_snapshot(self) -> dict[str, int]:
        return {
            k: int(self.catastrophe_failopen_reason.get(k, 0))
            for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS
        }

    def episode_turn_plan_catastrophe_reasons(
        self, start: dict[str, int]
    ) -> dict[str, int]:
        return {
            k: int(self.catastrophe_failopen_reason.get(k, 0))
            - int(start.get(k, 0))
            for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "fail_open": int(self.fail_open),
            "latencies_ms": list(self.latencies_ms),
            "failopen_reason": {
                k: int(self.failopen_reason.get(k, 0)) for k in COMBAT_JEV_FAILOPEN_REASONS
            },
            "turn_plan_turns": int(self.turn_plan_turns),
            "turn_plan_fulfilled": int(self.turn_plan_fulfilled),
            "turn_plan_catastrophe": int(self.turn_plan_catastrophe),
            "catastrophe_failopen_reason": {
                k: int(self.catastrophe_failopen_reason.get(k, 0))
                for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS
            },
            "turn_plan_error_samples": list(self.turn_plan_error_samples),
            "turn_plan_pruned_count": int(self.turn_plan_pruned_count),
            "turn_plan_heuristic_score_samples": list(
                self.turn_plan_heuristic_score_samples
            ),
            "turn_plan_replan_trigger": dict(self.turn_plan_replan_trigger),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CombatJevTelemetry":
        tel = cls()
        if not data:
            return tel
        tel.calls = int(data.get("calls") or 0)
        tel.fail_open = int(data.get("fail_open") or 0)
        tel.latencies_ms = [float(x) for x in (data.get("latencies_ms") or [])]
        reasons = data.get("failopen_reason") or {}
        for k in COMBAT_JEV_FAILOPEN_REASONS:
            tel.failopen_reason[k] = int(reasons.get(k, 0) or 0)
        tel.turn_plan_turns = int(data.get("turn_plan_turns") or 0)
        tel.turn_plan_fulfilled = int(data.get("turn_plan_fulfilled") or 0)
        tel.turn_plan_catastrophe = int(data.get("turn_plan_catastrophe") or 0)
        cat = data.get("catastrophe_failopen_reason") or {}
        for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS:
            tel.catastrophe_failopen_reason[k] = int(cat.get(k, 0) or 0)
        tel.turn_plan_error_samples = [
            str(x) for x in (data.get("turn_plan_error_samples") or [])
        ][:TURN_PLAN_ERROR_SAMPLE_MAX]
        tel.turn_plan_pruned_count = int(data.get("turn_plan_pruned_count") or 0)
        tel.turn_plan_heuristic_score_samples = [
            float(x) for x in (data.get("turn_plan_heuristic_score_samples") or [])
        ]
        triggers = data.get("turn_plan_replan_trigger") or {}
        if isinstance(triggers, dict):
            tel.turn_plan_replan_trigger = {
                str(k): int(v) for k, v in triggers.items()
            }
        return tel

    def merge(self, other: "CombatJevTelemetry") -> "CombatJevTelemetry":
        self.calls += int(other.calls)
        self.fail_open += int(other.fail_open)
        self.latencies_ms.extend(other.latencies_ms)
        for k in COMBAT_JEV_FAILOPEN_REASONS:
            self.failopen_reason[k] = int(self.failopen_reason.get(k, 0)) + int(
                other.failopen_reason.get(k, 0)
            )
        self.turn_plan_turns += int(other.turn_plan_turns)
        self.turn_plan_fulfilled += int(other.turn_plan_fulfilled)
        self.turn_plan_catastrophe += int(other.turn_plan_catastrophe)
        for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS:
            self.catastrophe_failopen_reason[k] = int(
                self.catastrophe_failopen_reason.get(k, 0)
            ) + int(other.catastrophe_failopen_reason.get(k, 0))
        for sample in other.turn_plan_error_samples:
            self.record_turn_plan_error_sample(sample)
        self.turn_plan_pruned_count += int(other.turn_plan_pruned_count)
        self.turn_plan_heuristic_score_samples.extend(
            other.turn_plan_heuristic_score_samples
        )
        if len(self.turn_plan_heuristic_score_samples) > 4096:
            self.turn_plan_heuristic_score_samples = (
                self.turn_plan_heuristic_score_samples[-4096:]
            )
        for key, count in other.turn_plan_replan_trigger.items():
            self.turn_plan_replan_trigger[key] = int(
                self.turn_plan_replan_trigger.get(key, 0)
            ) + int(count)
        return self

    def as_report(self, *, n_episodes: int = 0) -> dict[str, Any]:
        n = len(self.latencies_ms)
        p50 = p95 = mean = None
        if n:
            arr = np.sort(np.asarray(self.latencies_ms, dtype=float))
            p50 = round(float(np.percentile(arr, 50)), 2)
            p95 = round(float(np.percentile(arr, 95)), 2)
            mean = round(float(np.mean(arr)), 2)
        calls = int(self.calls)
        fail = int(self.fail_open)
        return {
            "jev_calls": calls,
            "jev_failopen": fail,
            "failopen_rate": round(fail / calls, 4) if calls else 0.0,
            "latency_ms": {"p50": p50, "p95": p95, "n": n, "mean": mean},
            "combat_jev_calls_mean": (
                round(calls / n_episodes, 4) if n_episodes else 0.0
            ),
            "jev_failopen_reason": {
                k: int(self.failopen_reason.get(k, 0)) for k in COMBAT_JEV_FAILOPEN_REASONS
            },
            **self._turn_plan_report_rates(),
            "note": (
                "Combat-Jev is an optional bypass (--combat-policy jev), not a "
                "hang swap. Default remains ppo/bh_v1. Conf min "
                f"{COMBAT_JEV_CONF_MIN}. "
                "jev-turn rates (jev_fulfilled_rate, catastrophe_failopen_rate) "
                "apply when --combat-policy jev-turn records turn_plan_turns."
            ),
        }


def _turn_plan_rates_from_rows(rows: list[dict]) -> dict[str, Any]:
    turns = int(sum(int(r.get("turn_plan_turns") or 0) for r in rows))
    fulfilled = int(sum(int(r.get("turn_plan_fulfilled") or 0) for r in rows))
    catastrophe = int(sum(int(r.get("turn_plan_catastrophe_failopen") or 0) for r in rows))
    reasons = {k: 0 for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS}
    for r in rows:
        for k in COMBAT_JEV_TURN_CATASTROPHE_REASONS:
            reasons[k] += int(r.get(f"turn_plan_catastrophe_{k}") or 0)
    return {
        "turn_plan_turns": turns,
        "turn_plan_fulfilled": fulfilled,
        "turn_plan_catastrophe_failopen": catastrophe,
        "jev_fulfilled_rate": round(fulfilled / turns, 4) if turns else 0.0,
        "catastrophe_failopen_rate": round(catastrophe / turns, 4) if turns else 0.0,
        "jev_turn_catastrophe_reason": reasons,
    }


def summarize_combat_jev(
    rows: list[dict],
    telemetry: CombatJevTelemetry | None = None,
    *,
    n_episodes: int | None = None,
) -> dict[str, Any]:
    n_eps = n_episodes if n_episodes is not None else len(rows)
    if telemetry is not None:
        return telemetry.as_report(n_episodes=n_eps)
    calls = int(sum(int(r.get("combat_jev_calls") or 0) for r in rows))
    fail = int(sum(int(r.get("combat_jev_fail_open") or 0) for r in rows))
    reasons = {k: 0 for k in COMBAT_JEV_FAILOPEN_REASONS}
    for r in rows:
        for k in COMBAT_JEV_FAILOPEN_REASONS:
            reasons[k] += int(r.get(f"combat_jev_failopen_{k}") or 0)
    return {
        "jev_calls": calls,
        "jev_failopen": fail,
        "failopen_rate": round(fail / calls, 4) if calls else 0.0,
        "latency_ms": {"p50": None, "p95": None, "n": 0, "mean": None},
        "combat_jev_calls_mean": round(calls / n_eps, 4) if n_eps else 0.0,
        "jev_failopen_reason": reasons,
        **_turn_plan_rates_from_rows(rows),
        "note": (
            "Combat-Jev is an optional bypass (--combat-policy jev), not a "
            "hang swap. Default remains ppo/bh_v1."
        ),
    }


def _choice_question(options: list[tuple[str, int, str]]) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": COMBAT_JEV_INSTRUCTIONS,
        "criteria": {action_id: summary for action_id, _idx, summary in options},
    }


def choose_combat_step(
    combat: CombatState,
    combat_mask: np.ndarray,
    rng: np.random.RandomState,
    combat_model: Any,
    *,
    adapter: Any,
    combat_obs: np.ndarray | None = None,
    telemetry: CombatJevTelemetry | None = None,
    min_conf: float = COMBAT_JEV_CONF_MIN,
    owner: Creature | None = None,
) -> tuple[int, dict[str, Any]]:
    """Pick a local combat action via ``combat_step_choice``, else fail-open.

    Returns ``(local_action, shadow_fields)``. Never skips the decision.
    """
    obs = combat_obs if combat_obs is not None else encode_observation(combat)
    mask = np.asarray(combat_mask)
    end_turn_legal = int(mask[ACTION_END_TURN]) == 1 if mask.size > ACTION_END_TURN else False

    def _open(reason: str, pick: JevAnswer, latency_ms: float = 0.0) -> tuple[int, dict[str, Any]]:
        if telemetry is not None:
            telemetry.record(latency_ms, fail_reason=reason)
        local = fail_open_local(combat_model, obs, mask, rng)
        log = pick.as_log()
        log["shadow_decision"] = CHOICE_COMBAT_STEP
        log["jev_failopen_reason"] = reason
        log["combat_jev_failopen"] = True
        log["executed_id"] = combat_action_id(local)
        return local, log

    if combat.pending_choice is not None:
        local = fail_open_local(combat_model, obs, mask, rng)
        log = JevAnswer(status="skipped", fallback_reason="pending_choice").as_log()
        log["shadow_decision"] = CHOICE_COMBAT_STEP
        log["executed_id"] = combat_action_id(local)
        return local, log

    options = enumerate_legal_combat_actions(combat, mask, owner=owner)
    id_to_idx = {action_id: idx for action_id, idx, _summary in options}
    if not options:
        pick = JevAnswer(status="error", fallback_reason=FAILOPEN_EMPTY_LIST)
        return _open(FAILOPEN_EMPTY_LIST, pick, 0.0)

    state = compress_combat_state(combat, owner=owner, end_turn_legal=end_turn_legal)
    questions = {CHOICE_COMBAT_STEP: _choice_question(options)}
    t0 = time.perf_counter()
    try:
        answers = adapter.system_one(state, questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except JevError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reason = classify_jev_error(e)
        pick = JevAnswer(status="error", fallback_reason=reason)
        return _open(reason, pick, latency_ms)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reason = classify_jev_error(e)
        pick = JevAnswer(status="error", fallback_reason=reason)
        return _open(reason, pick, latency_ms)

    raw = answers.get(CHOICE_COMBAT_STEP) if isinstance(answers, dict) else None
    if raw is None:
        pick = JevAnswer(status="error", fallback_reason=FAILOPEN_ERROR)
        return _open(FAILOPEN_ERROR, pick, latency_ms)
    pick = apply_choice_confidence(raw, min_conf=min_conf)
    choice_id = pick.choice
    if choice_id not in id_to_idx:
        pick.status = "error"
        pick.fallback_reason = FAILOPEN_BAD_ID
        return _open(FAILOPEN_BAD_ID, pick, latency_ms)
    if pick.status != "ok":
        return _open(FAILOPEN_LOW_CONF, pick, latency_ms)

    local = id_to_idx[choice_id]
    if telemetry is not None:
        telemetry.record(latency_ms, fail_reason=None)
    log = pick.as_log()
    log["shadow_decision"] = CHOICE_COMBAT_STEP
    log["combat_jev_failopen"] = False
    log["executed_id"] = combat_action_id(local)
    log["jev_choice_id"] = CHOICE_COMBAT_STEP
    return local, log


__all__ = [
    "CHOICE_COMBAT_STEP",
    "COMBAT_JEV_CONF_MIN",
    "COMBAT_JEV_FAILOPEN_REASONS",
    "COMBAT_JEV_HAND_TRUNCATE",
    "COMBAT_JEV_INSTRUCTIONS",
    "COMBAT_JEV_MONSTER_TRUNCATE",
    "COMBAT_JEV_TURN_CATASTROPHE_REASONS",
    "CombatJevTelemetry",
    "FAILOPEN_BAD_ID",
    "FAILOPEN_EMPTY_LIST",
    "FAILOPEN_ERROR",
    "FAILOPEN_ILLEGAL_PLAN",
    "FAILOPEN_LOW_CONF",
    "FAILOPEN_MISSING_HUNG_PPO",
    "FAILOPEN_REPLAN_CAP",
    "FAILOPEN_TIMEOUT",
    "TURN_PLAN_ERROR_SAMPLE_MAX",
    "choose_combat_step",
    "classify_jev_error",
    "combat_action_id",
    "compress_combat_state",
    "enumerate_legal_combat_actions",
    "fail_open_bh_v1_required",
    "fail_open_local",
    "format_turn_plan_error_sample",
    "hung_ppo_local",
    "parse_combat_action_id",
    "summarize_combat_action",
    "summarize_combat_jev",
]
