"""Legal RunEnv candidate extraction and Jev-backed non-combat selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sts2_env.eval.jev import (
    CONTENT_MAP_REF,
    HP_PRESSURE_SCORE_CRITERIA,
    PLUS_CARD_CRITERION,
    JevAnswer,
    JevClient,
    JevError,
    apply_choice_confidence,
    local_hp_pressure,
    rest_or_continue_override,
)
from sts2_env.gym_env.run_env import (
    STS2RunEnv,
    _BOSS_RELIC_SIZE,
    _BOSS_RELIC_START,
    _CARD_RWD_EXTRA_START,
    _CARD_RWD_REROLL,
    _CARD_RWD_START,
    _COMBAT_SIZE,
    _COMBAT_START,
    _EVENT_SIZE,
    _EVENT_START,
    _MAP_SIZE,
    _MAP_START,
    _REST_SIZE,
    _REST_START,
    _SHOP_SIZE,
    _SHOP_START,
    _TREASURE_START,
)
from sts2_env.run.run_manager import RunManager

logger = logging.getLogger(__name__)

DECISION_MAP_FORK = "map_fork"
DECISION_REST_OR_CONTINUE = "rest_or_continue"
DECISION_CARD_REWARD = "card_reward"
DECISION_REST_SITE = "rest_site"
DECISION_SIMILAR = "similar"
DECISION_NONE = "none"

REST_POINT_TYPES = frozenset({"REST_SITE", "RestSite", "restsite"})


@dataclass
class Candidate:
    key: str
    run_action: int
    description: str
    legal: bool
    visible: bool = True
    kind: str = ""
    upgraded: bool = False
    is_rest: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


def _mgr(env: STS2RunEnv) -> RunManager:
    mgr = getattr(env, "_mgr", None)
    if mgr is None:
        raise RuntimeError("env has no run manager")
    return mgr


def collect_candidates(env: STS2RunEnv, mask: np.ndarray) -> list[Candidate]:
    """Pair RunEnv legal mask slots with RunManager actions. Invisible/illegal stay marked."""
    mgr = _mgr(env)
    mask = np.asarray(mask)
    phase = mgr.phase
    actions = mgr.get_available_actions()
    cands: list[Candidate] = []

    def _legal(idx: int) -> bool:
        return 0 <= idx < len(mask) and int(mask[idx]) == 1

    if phase != RunManager.PHASE_COMBAT and any(
        a.get("action") in {"choose", "confirm_choice"} for a in actions
    ):
        if any(a.get("action") == "confirm_choice" for a in actions):
            cands.append(
                Candidate(
                    key="confirm",
                    run_action=_COMBAT_START,
                    description="Confirm current multi-select choice",
                    legal=_legal(_COMBAT_START),
                    kind="confirm",
                )
            )
        choose_actions = [a for a in actions if a.get("action") == "choose"]
        for i, act in enumerate(choose_actions[: max(_COMBAT_SIZE - 1, 0)]):
            idx = _COMBAT_START + 1 + i
            card_id = act.get("card_id", i)
            cands.append(
                Candidate(
                    key=f"choose_{i}",
                    run_action=idx,
                    description=f"Choose option {i} ({card_id})",
                    legal=_legal(idx),
                    kind="choose",
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_MAP_CHOICE:
        for i, act in enumerate(actions[:_MAP_SIZE]):
            idx = _MAP_START + i
            point_type = str(act.get("point_type", "UNKNOWN"))
            visible = point_type not in {"UNASSIGNED", ""}
            cands.append(
                Candidate(
                    key=f"map_{i}",
                    run_action=idx,
                    description=f"Map node {i}: {point_type} at {act.get('coord')}",
                    legal=_legal(idx),
                    visible=visible,
                    kind="map",
                    is_rest=point_type in REST_POINT_TYPES,
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_CARD_REWARD:
        if any(a.get("action") == "pick_potion" for a in actions):
            cands.append(
                Candidate(
                    key="potion_take",
                    run_action=_CARD_RWD_START,
                    description="Take potion reward",
                    legal=_legal(_CARD_RWD_START),
                    kind="potion",
                )
            )
            cands.append(
                Candidate(
                    key="potion_skip",
                    run_action=_CARD_RWD_START + 3,
                    description="Skip potion reward",
                    legal=_legal(_CARD_RWD_START + 3),
                    kind="potion",
                )
            )
            return cands
        if any(a.get("action") == "pick_relic_reward" for a in actions):
            cands.append(
                Candidate(
                    key="relic_take",
                    run_action=_CARD_RWD_START,
                    description="Take relic reward",
                    legal=_legal(_CARD_RWD_START),
                    kind="relic",
                )
            )
            cands.append(
                Candidate(
                    key="relic_skip",
                    run_action=_CARD_RWD_START + 3,
                    description="Skip relic reward",
                    legal=_legal(_CARD_RWD_START + 3),
                    kind="relic",
                )
            )
            return cands
        pick_actions = [a for a in actions if a.get("action") == "pick_card"]
        for i, act in enumerate(pick_actions):
            if i < 3:
                idx = _CARD_RWD_START + i
            else:
                idx = _CARD_RWD_EXTRA_START + (i - 3)
            upgraded = bool(act.get("upgraded"))
            card_id = act.get("card_id", i)
            desc = f"Pick card {i}: {card_id}"
            if upgraded:
                desc += " (upgraded/+; not a natural Act1 reward drop)"
            cands.append(
                Candidate(
                    key=f"card_{i}",
                    run_action=idx,
                    description=desc,
                    legal=_legal(idx),
                    kind="card",
                    upgraded=upgraded,
                    payload=dict(act),
                )
            )
        skip_idx = _CARD_RWD_START + 3
        cands.append(
            Candidate(
                key="card_skip",
                run_action=skip_idx,
                description="Skip card reward",
                legal=_legal(skip_idx),
                kind="skip",
            )
        )
        if any(a.get("action") == "reroll_card_reward" for a in actions):
            cands.append(
                Candidate(
                    key="card_reroll",
                    run_action=_CARD_RWD_REROLL,
                    description="Reroll card reward",
                    legal=_legal(_CARD_RWD_REROLL),
                    kind="reroll",
                )
            )
        return cands

    if phase == RunManager.PHASE_REST_SITE:
        rest_actions = [a for a in actions if a.get("action") == "rest_option"]
        for i, act in enumerate(rest_actions[:_REST_SIZE]):
            idx = _REST_START + i
            option_id = str(act.get("option_id", i))
            cands.append(
                Candidate(
                    key=f"rest_{option_id}",
                    run_action=idx,
                    description=f"Rest option {option_id}: {act.get('label', option_id)}",
                    legal=_legal(idx) and bool(act.get("enabled", True)),
                    visible=bool(act.get("enabled", True)),
                    kind="rest_option",
                    is_rest=option_id in {"HEAL", "heal", "Rest"},
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_BOSS_RELIC:
        relics = [a for a in actions if a.get("action") == "pick_relic"]
        for i, act in enumerate(relics[:_BOSS_RELIC_SIZE]):
            idx = _BOSS_RELIC_START + i
            cands.append(
                Candidate(
                    key=f"boss_relic_{i}",
                    run_action=idx,
                    description=f"Boss relic {i}: {act.get('relic_id', i)}",
                    legal=_legal(idx),
                    kind="boss_relic",
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_SHOP:
        leave = _SHOP_START
        cands.append(
            Candidate(
                key="shop_leave",
                run_action=leave,
                description="Leave shop",
                legal=_legal(leave),
                kind="shop",
            )
        )
        buyable = [a for a in actions if a.get("action") != "leave_shop"]
        for i, act in enumerate(buyable[: _SHOP_SIZE - 1]):
            idx = _SHOP_START + 1 + i
            cands.append(
                Candidate(
                    key=f"shop_buy_{i}",
                    run_action=idx,
                    description=f"Shop buy {act.get('action', i)}",
                    legal=_legal(idx),
                    kind="shop",
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_EVENT:
        event_actions = [a for a in actions if a.get("action") == "event_choice"]
        for i, act in enumerate(event_actions[:_EVENT_SIZE]):
            idx = _EVENT_START + i
            enabled = bool(act.get("enabled", True))
            cands.append(
                Candidate(
                    key=f"event_{act.get('option_id', i)}",
                    run_action=idx,
                    description=f"Event {act.get('label', act.get('option_id', i))}",
                    legal=_legal(idx) and enabled,
                    visible=enabled,
                    kind="event",
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_TREASURE:
        cands.append(
            Candidate(
                key="treasure_collect",
                run_action=_TREASURE_START,
                description="Collect treasure",
                legal=_legal(_TREASURE_START),
                kind="treasure",
            )
        )
        return cands

    return cands


def strip_illegal_invisible(cands: list[Candidate]) -> list[Candidate]:
    return [c for c in cands if c.legal and c.visible]


def classify_decision(phase: str, cands: list[Candidate]) -> str:
    if phase == RunManager.PHASE_MAP_CHOICE:
        has_rest = any(c.is_rest for c in cands)
        has_continue = any(not c.is_rest for c in cands)
        if has_rest and has_continue:
            return DECISION_REST_OR_CONTINUE
        return DECISION_MAP_FORK
    if phase == RunManager.PHASE_CARD_REWARD:
        return DECISION_CARD_REWARD
    if phase == RunManager.PHASE_REST_SITE:
        return DECISION_REST_SITE
    if phase in {
        RunManager.PHASE_EVENT,
        RunManager.PHASE_SHOP,
        RunManager.PHASE_BOSS_RELIC,
    }:
        return DECISION_SIMILAR
    if cands:
        return DECISION_SIMILAR
    return DECISION_NONE


def _run_state_blob(mgr: RunManager) -> dict[str, Any]:
    rs = mgr.run_state
    player = rs.player
    return {
        "content_map": CONTENT_MAP_REF,
        "act": rs.current_act_index,
        "floor": rs.total_floor,
        "hp": player.current_hp,
        "max_hp": player.max_hp,
        "gold": player.gold,
        "deck_size": len(player.deck),
        "relics": len(rs.relics),
        "phase": mgr.phase,
        "hp_ratio": player.current_hp / max(player.max_hp, 1),
    }


def _choice_question(cands: list[Candidate], instructions: str) -> dict[str, Any]:
    criteria: dict[str, str] = {}
    for c in cands:
        desc = c.description
        if c.upgraded:
            desc = f"{desc}. {PLUS_CARD_CRITERION}"
        criteria[c.key] = desc
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }


def _lookup(cands: list[Candidate], key: str | None) -> Candidate | None:
    if key is None:
        return None
    for c in cands:
        if c.key == key:
            return c
    return None


def _legal_random_from(cands: list[Candidate], rng: np.random.RandomState) -> int:
    if not cands:
        return 0
    pick = cands[int(rng.randint(0, len(cands)))]
    return pick.run_action


def choose_jev_noncombat(
    env: STS2RunEnv,
    mask: np.ndarray,
    rng: np.random.RandomState,
    adapter: JevClient,
) -> tuple[int, dict[str, Any]]:
    """Pick a legal non-combat RunEnv action via Jev, or random on fallback.

    Errors are logged on the returned shadow fields (status=error) and the
    action falls back to legal random. The decision point is never skipped.
    """
    mgr = _mgr(env)
    all_cands = collect_candidates(env, mask)
    cands = strip_illegal_invisible(all_cands)
    if not cands:
        valid = np.flatnonzero(np.asarray(mask) == 1)
        action = int(rng.choice(valid)) if valid.size else 0
        logger.error("Jev status=error fallback=no_legal_visible_candidates; action=%s", action)
        log = JevAnswer(
            status="error",
            fallback_reason="no_legal_visible_candidates",
        ).as_log()
        return action, log

    decision = classify_decision(mgr.phase, cands)
    state = _run_state_blob(mgr)
    state["decision"] = decision
    state["candidates"] = [
        {"key": c.key, "description": c.description, "upgraded": c.upgraded, "is_rest": c.is_rest}
        for c in cands
    ]

    try:
        action, answer = _decide(decision, cands, state, adapter, mgr, rng)
    except JevError as e:
        logger.error("Jev %s error; falling back to legal random: %s", decision, e)
        action = _legal_random_from(cands, rng)
        answer = JevAnswer(status="error", fallback_reason=str(e))
    log = answer.as_log()
    log["shadow_decision"] = decision
    if answer.status == "error":
        logger.error(
            "Jev %s status=error fallback=%s; action=%s",
            decision,
            answer.fallback_reason,
            action,
        )
    elif answer.status == "uncertain":
        logger.warning(
            "Jev %s status=uncertain fallback=%s; action=%s",
            decision,
            answer.fallback_reason,
            action,
        )
    return action, log


def _decide(
    decision: str,
    cands: list[Candidate],
    state: dict[str, Any],
    adapter: JevClient,
    mgr: RunManager,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer]:
    if decision == DECISION_REST_OR_CONTINUE:
        return _decide_rest_or_continue(cands, state, adapter, mgr, rng)
    instructions = _instructions_for(decision)
    questions = {
        "pick": _choice_question(
            cands,
            instructions,
        )
    }
    answers = adapter.system_one(state, questions)
    pick = apply_choice_confidence(answers.get("pick") or JevAnswer(status="error"))
    if pick.status != "ok":
        return _legal_random_from(cands, rng), pick
    chosen = _lookup(cands, pick.choice)
    if chosen is None:
        pick.status = "error"
        pick.fallback_reason = f"choice {pick.choice!r} not in legal candidates"
        return _legal_random_from(cands, rng), pick
    return chosen.run_action, pick


def _decide_rest_or_continue(
    cands: list[Candidate],
    state: dict[str, Any],
    adapter: JevClient,
    mgr: RunManager,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer]:
    rest_cands = [c for c in cands if c.is_rest]
    continue_cands = [c for c in cands if not c.is_rest]
    questions = {
        "hp_pressure": {
            "type": "score",
            "instructions": (
                "Score current HP pressure for an Act1 map fork. "
                f"Reference {CONTENT_MAP_REF}. Higher means rest is more urgent."
            ),
            "criteria": HP_PRESSURE_SCORE_CRITERIA,
        },
        "pick": _choice_question(
            cands,
            (
                "Choose the next Act1 map node (rest vs continue). "
                f"See {CONTENT_MAP_REF}."
            ),
        ),
    }
    answers = adapter.system_one(state, questions)
    pressure_ans = answers.get("hp_pressure") or JevAnswer(status="error")
    pick = apply_choice_confidence(answers.get("pick") or JevAnswer(status="error"))

    hp = int(mgr.run_state.player.current_hp)
    max_hp = int(mgr.run_state.player.max_hp)
    if pressure_ans.status == "ok" and pressure_ans.score is not None:
        pressure = float(pressure_ans.score)
        pressure_source = "jev_score"
    else:
        pressure = local_hp_pressure(hp, max_hp)
        pressure_source = "local_fallback"
        if pressure_ans.status != "ok":
            # Score failed: still decide; log on the pick answer.
            pass

    override = rest_or_continue_override(pressure)
    merged = pick
    merged.score = pressure
    if override == "rest" and rest_cands:
        merged.status = "ok"
        merged.choice = rest_cands[0].key
        merged.fallback_reason = f"hp_pressure {pressure:.2f} >= 2.0 prefer rest ({pressure_source})"
        return rest_cands[0].run_action, merged
    if override == "continue" and continue_cands:
        # Continue pool: still require Choice among continue nodes when confident.
        cont_questions = {
            "pick": _choice_question(
                continue_cands,
                f"Choose a non-rest Act1 map node. See {CONTENT_MAP_REF}.",
            )
        }
        try:
            cont_answers = adapter.system_one(state, cont_questions)
            cont_pick = apply_choice_confidence(
                cont_answers.get("pick") or JevAnswer(status="error")
            )
        except JevError as e:
            cont_pick = JevAnswer(status="error", fallback_reason=str(e))
        cont_pick.score = pressure
        if cont_pick.status == "ok":
            chosen = _lookup(continue_cands, cont_pick.choice)
            if chosen is not None:
                cont_pick.fallback_reason = (
                    f"hp_pressure {pressure:.2f} <= 1.0 prefer continue ({pressure_source})"
                )
                return chosen.run_action, cont_pick
        action = _legal_random_from(continue_cands, rng)
        cont_pick.status = cont_pick.status if cont_pick.status != "ok" else "error"
        if not cont_pick.fallback_reason:
            cont_pick.fallback_reason = (
                f"hp_pressure {pressure:.2f} prefer continue; choice fallback ({pressure_source})"
            )
        return action, cont_pick

    if pick.status != "ok":
        return _legal_random_from(cands, rng), pick
    chosen = _lookup(cands, pick.choice)
    if chosen is None:
        pick.status = "error"
        pick.fallback_reason = f"choice {pick.choice!r} not in legal candidates"
        return _legal_random_from(cands, rng), pick
    pick.score = pressure
    return chosen.run_action, pick


def _instructions_for(decision: str) -> str:
    base = f"Act1 Ironclad run. Criteria: {CONTENT_MAP_REF}. "
    if decision == DECISION_MAP_FORK:
        return base + "Choose the next map node among legal visible forks."
    if decision == DECISION_CARD_REWARD:
        return (
            base
            + "Choose a card reward or skip. "
            + PLUS_CARD_CRITERION
        )
    if decision == DECISION_REST_SITE:
        return base + "Choose a rest-site option (heal vs smith vs relic options)."
    return base + "Choose among legal visible options at this decision point."
