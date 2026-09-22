"""Legal RunEnv candidate extraction and Jev-backed non-combat selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sts2_env.eval.jev import (
    CARD_FIT_ASSIST_MIN,
    CARD_FIT_ASSIST_REASON,
    CARD_FIT_SCORE_CRITERIA,
    CHOICE_EVENT,
    CHOICE_NEOW_BOON,
    CONTENT_MAP_REF,
    DEFAULT_JEV_PHASES,
    EVENT_CHOICE_INSTRUCTIONS,
    EVENT_OPTIONS_EMPTY_REASON,
    HP_PRESSURE_REST,
    HP_PRESSURE_SCORE_CRITERIA,
    JEV_EVENT_OFF_REASON,
    JEV_NEOW_OFF_REASON,
    NEOW_OPTIONS_EMPTY_REASON,
    JEV_PHASE_TOKENS,
    NEOW_BOON_INSTRUCTIONS,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    NON_JEV_PHASE_REASON,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
    SHOP_RANDOM_REASON,
    UNKNOWN_DEFER_CONF,
    UNKNOWN_DEFERRED_REASON,
    UNKNOWN_MAP_CRITERION,
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
from sts2_env.run.run_manager import NEOW_EVENT_ID, RunManager

logger = logging.getLogger(__name__)

DECISION_MAP_FORK = "map_fork"
DECISION_REST_OR_CONTINUE = "rest_or_continue"
DECISION_CARD_REWARD = "card_reward"
DECISION_POTION_OR_RELIC = "potion_or_relic_reward"
DECISION_REST_SITE = "rest_site"
DECISION_EVENT = "event_choice"
DECISION_NEOW = "neow_boon"
DECISION_SHOP = "shop"
DECISION_SIMILAR = "similar"
DECISION_NONE = "none"

REST_POINT_TYPES = frozenset({"REST_SITE", "RestSite", "restsite"})
UNKNOWN_POINT_TYPES = frozenset({"UNKNOWN", "Unknown", "unknown"})
POTION_OR_RELIC_ACTIONS = frozenset({"pick_potion", "pick_relic_reward"})
SKIP_ACTIONS = frozenset({"skip"})
PENDING_CHOICE_ACTIONS = frozenset({"choose", "confirm_choice"})


@dataclass(frozen=True)
class JevPolicyFlags:
    """Which non-combat phases call Jev. Default preserves MAP/REST/CARD tables."""

    phases: frozenset[str] = DEFAULT_JEV_PHASES
    event: bool = False
    neow: bool | None = None

    def resolved_phases(self) -> frozenset[str]:
        phases = set(self.phases)
        if self.event:
            phases.add("event")
        return frozenset(phases)

    def allows_event(self) -> bool:
        return "event" in self.resolved_phases()

    def allows_neow(self) -> bool:
        if self.neow is None:
            return self.allows_event()
        return bool(self.neow)


DEFAULT_JEV_FLAGS = JevPolicyFlags()


def parse_jev_phases(text: str | None) -> frozenset[str]:
    raw = (text or "").strip()
    if not raw:
        return DEFAULT_JEV_PHASES
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in JEV_PHASE_TOKENS]
    if bad:
        raise SystemExit(f"unknown --jev-phases token(s): {bad}; expected {sorted(JEV_PHASE_TOKENS)}")
    return frozenset(parts)


def resolve_jev_flags(
    *,
    jev_event: str = "off",
    jev_phases: str | None = None,
    jev_neow: str | None = None,
) -> JevPolicyFlags:
    phases = parse_jev_phases(jev_phases)
    event = jev_event == "on" or "event" in phases
    if event:
        phases = phases | {"event"}
    neow: bool | None
    if jev_neow is None:
        neow = None
    else:
        neow = jev_neow == "on"
    return JevPolicyFlags(phases=phases, event=event, neow=neow)


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
    is_unknown: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


def _mgr(env: STS2RunEnv) -> RunManager:
    mgr = getattr(env, "_mgr", None)
    if mgr is None:
        raise RuntimeError("env has no run manager")
    return mgr


def is_potion_or_relic_reward(actions: list[dict[str, Any]]) -> bool:
    """Potion/relic screens share PHASE_CARD_REWARD but are not pick_card."""
    return any(a.get("action") in POTION_OR_RELIC_ACTIONS for a in actions)


def _try_preview_card(card_id: Any, upgraded: bool):
    try:
        from sts2_env.cards.factory import create_card
        from sts2_env.core.enums import CardId
    except Exception:
        return None
    raw = str(card_id)
    cid = None
    if raw in CardId.__members__:
        cid = CardId[raw]
    else:
        for suffix in ("_IRONCLAD", "_CARD", "_STATUS"):
            cand = raw + suffix
            if cand in CardId.__members__:
                cid = CardId[cand]
                break
    if cid is None:
        return None
    try:
        return create_card(cid, upgraded=upgraded)
    except Exception:
        return None


def card_blurb(act: dict[str, Any]) -> str:
    """Short label for a pick_card offer (cost / type / dmg-block / plus note)."""
    card_id = act.get("card_id", "?")
    upgraded = bool(act.get("upgraded"))
    rarity = str(act.get("rarity") or "")
    plus_note = ""
    name = str(card_id).replace("_", " ")
    bits: list[str] = []
    card = _try_preview_card(card_id, upgraded)
    if card is not None:
        name = card.card_id.name.replace("_", " ").title()
        if upgraded or card.upgraded:
            name += "+"
            plus_note = " (Smith/Neow upgraded; not a natural Act1 drop)"
        if rarity:
            bits.append(rarity.lower())
        bits.append(card.card_type.name.lower())
        cost = "X" if getattr(card, "has_energy_cost_x", False) else str(card.cost)
        bits.append(f"cost {cost}")
        if card.base_damage:
            bits.append(f"{card.base_damage} dmg")
        block = (card.effect_vars or {}).get("block", card.base_block)
        if block:
            bits.append(f"{block} block")
        for k, v in (card.effect_vars or {}).items():
            if k == "block":
                continue
            bits.append(f"{k} {v}")
    else:
        if upgraded:
            name += "+"
            plus_note = " (Smith/Neow upgraded; not a natural Act1 drop)"
        if rarity:
            bits.append(rarity.lower())
    core = name
    if bits:
        core += " — " + ", ".join(bits)
    return core + plus_note


def build_rest_options(
    actions: list[dict[str, Any]],
    legal_fn,
) -> list[Candidate]:
    """REST_SITE options: ``rest_option`` → ``_REST_START+i``; pending → combat slots.

    Pending ``confirm_choice`` → ``_COMBAT_START``; ``choose`` index i →
    ``_COMBAT_START+1+i`` (same as EVENT / ``run_env`` non-combat choice).
    Keys are ``rest_confirm`` / ``rest_choose_{i}``.
    """
    cands: list[Candidate] = []
    pending = any(a.get("action") in PENDING_CHOICE_ACTIONS for a in actions)
    if pending:
        if any(a.get("action") == "confirm_choice" for a in actions):
            cands.append(
                Candidate(
                    key="rest_confirm",
                    run_action=_COMBAT_START,
                    description="Confirm current rest-site multi-select choice",
                    legal=legal_fn(_COMBAT_START),
                    kind="confirm",
                    payload={"option_id": "confirm", "list_index": -1},
                )
            )
        choose_actions = [a for a in actions if a.get("action") == "choose"]
        for i, act in enumerate(choose_actions[: max(_COMBAT_SIZE - 1, 0)]):
            idx = _COMBAT_START + 1 + i
            label = str(act.get("label") or act.get("card_id") or i)
            desc_txt = str(act.get("description") or "")
            blurb = f"{label}: {desc_txt}".rstrip(": ")
            cands.append(
                Candidate(
                    key=f"rest_choose_{i}",
                    run_action=idx,
                    description=blurb,
                    legal=legal_fn(idx),
                    kind="choose",
                    payload={**dict(act), "list_index": i},
                )
            )
        return cands

    rest_actions = [a for a in actions if a.get("action") == "rest_option"]
    for i, act in enumerate(rest_actions[:_REST_SIZE]):
        idx = _REST_START + i
        option_id = str(act.get("option_id", i))
        enabled = bool(act.get("enabled", True))
        cands.append(
            Candidate(
                key=f"rest_{option_id}",
                run_action=idx,
                description=f"Rest option {option_id}: {act.get('label', option_id)}",
                legal=legal_fn(idx) and enabled,
                visible=enabled,
                kind="rest_option",
                is_rest=option_id in {"HEAL", "heal", "Rest"},
                payload=dict(act),
            )
        )
    return cands


def build_options(
    actions: list[dict[str, Any]],
    legal_fn,
    *,
    phase: str | None = None,
) -> list[Candidate]:
    """True pick_card options, or REST_SITE rest_option / pending combat slots.

    Skip is included only when ``action==skip``. Upgraded offers are labelled
    as Smith/Neow, not natural Act1 drops. ``phase=REST_SITE`` maps pending
    ``choose`` / ``confirm_choice`` to combat slots (same as EVENT).
    """
    if phase == RunManager.PHASE_REST_SITE:
        return build_rest_options(actions, legal_fn)
    cands: list[Candidate] = []
    pick_actions = [a for a in actions if a.get("action") == "pick_card"]
    for i, act in enumerate(pick_actions):
        if i < 3:
            idx = _CARD_RWD_START + i
        else:
            idx = _CARD_RWD_EXTRA_START + (i - 3)
        upgraded = bool(act.get("upgraded"))
        blurb = card_blurb(act)
        desc = f"Pick card {i}: {blurb}"
        cands.append(
            Candidate(
                key=f"card_{i}",
                run_action=idx,
                description=desc,
                legal=legal_fn(idx),
                kind="card",
                upgraded=upgraded,
                payload=dict(act),
            )
        )
    if any(a.get("action") in SKIP_ACTIONS for a in actions):
        skip_idx = _CARD_RWD_START + 3
        cands.append(
            Candidate(
                key="card_skip",
                run_action=skip_idx,
                description="Skip card reward",
                legal=legal_fn(skip_idx),
                kind="skip",
            )
        )
    if any(a.get("action") == "reroll_card_reward" for a in actions):
        cands.append(
            Candidate(
                key="card_reroll",
                run_action=_CARD_RWD_REROLL,
                description="Reroll card reward",
                legal=legal_fn(_CARD_RWD_REROLL),
                kind="reroll",
            )
        )
    return cands


def _event_id_from_mgr(mgr: Any) -> str:
    event = getattr(mgr, "_event_model", None)
    if event is None:
        return ""
    return str(getattr(event, "event_id", "") or "")


def is_neow_or_boon_screen(event_id: str, actions: list[dict[str, Any]] | None = None) -> bool:
    """True when the EVENT screen is Neow / a boon. Silent skip otherwise."""
    eid = str(event_id or "")
    if eid == NEOW_EVENT_ID or eid.lower() == "neow":
        return True
    tokens = [eid]
    for act in actions or []:
        for key in ("event_id", "option_id", "label", "id"):
            val = act.get(key)
            if val:
                tokens.append(str(val))
    blob = " ".join(tokens).lower()
    return "neow" in blob or "boon" in blob


def build_event_options(
    actions: list[dict[str, Any]],
    legal_fn,
    *,
    event_id: str = "",
    is_neow: bool = False,
) -> list[Candidate]:
    """EVENT options: ``event_choice`` → ``_EVENT_START+i``; pending → combat slots.

    Pending ``confirm_choice`` → ``_COMBAT_START``; ``choose`` index i →
    ``_COMBAT_START+1+i`` (same as ``run_env._step_event``). ``enabled=False``
    options are not built.
    """
    cands: list[Candidate] = []
    pending = any(a.get("action") in PENDING_CHOICE_ACTIONS for a in actions)
    if pending:
        if any(a.get("action") == "confirm_choice" for a in actions):
            cands.append(
                Candidate(
                    key="confirm",
                    run_action=_COMBAT_START,
                    description="Confirm current event multi-select choice",
                    legal=legal_fn(_COMBAT_START),
                    kind="confirm",
                    payload={
                        "event_id": event_id,
                        "option_id": "confirm",
                        "list_index": -1,
                        "is_neow": is_neow,
                    },
                )
            )
        choose_actions = [a for a in actions if a.get("action") == "choose"]
        for i, act in enumerate(choose_actions[: max(_COMBAT_SIZE - 1, 0)]):
            idx = _COMBAT_START + 1 + i
            label = str(act.get("label") or act.get("card_id") or i)
            desc_txt = str(act.get("description") or "")
            option_id = act.get("option_id")
            key = str(option_id) if option_id not in (None, "") else f"event_{i}_{label}"
            blurb = f"{label}: {desc_txt}".rstrip(": ")
            cands.append(
                Candidate(
                    key=key,
                    run_action=idx,
                    description=blurb,
                    legal=legal_fn(idx),
                    kind="choose",
                    payload={
                        **dict(act),
                        "event_id": event_id,
                        "option_id": option_id or key,
                        "list_index": i,
                        "is_neow": is_neow,
                    },
                )
            )
        return cands

    event_actions = [
        a
        for a in actions
        if a.get("action") == "event_choice" and a.get("enabled") is not False
    ]
    for i, act in enumerate(event_actions[:_EVENT_SIZE]):
        if act.get("enabled") is False:
            continue
        idx = _EVENT_START + i
        label = str(act.get("label") or act.get("option_id") or i)
        desc_txt = str(act.get("description") or "")
        option_id = act.get("option_id")
        key = str(option_id) if option_id not in (None, "") else f"event_{i}_{label}"
        enabled = bool(act.get("enabled", True))
        cands.append(
            Candidate(
                key=key,
                run_action=idx,
                description=f"{label}: {desc_txt}".rstrip(": "),
                legal=legal_fn(idx) and enabled,
                visible=enabled,
                kind="event",
                payload={
                    **dict(act),
                    "event_id": event_id,
                    "option_id": option_id or key,
                    "list_index": i,
                    "is_neow": is_neow,
                },
            )
        )
    return cands


def collect_candidates(env: STS2RunEnv, mask: np.ndarray) -> list[Candidate]:
    """Pair RunEnv legal mask slots with RunManager actions. Invisible/illegal stay marked."""
    mgr = _mgr(env)
    mask = np.asarray(mask)
    phase = mgr.phase
    actions = mgr.get_available_actions()
    cands: list[Candidate] = []

    def _legal(idx: int) -> bool:
        return 0 <= idx < len(mask) and int(mask[idx]) == 1

    if phase == RunManager.PHASE_EVENT:
        event_id = _event_id_from_mgr(mgr)
        is_neow = is_neow_or_boon_screen(event_id, actions)
        return build_event_options(
            actions, _legal, event_id=event_id, is_neow=is_neow
        )

    if phase == RunManager.PHASE_REST_SITE:
        return build_options(actions, _legal, phase=RunManager.PHASE_REST_SITE)

    if phase != RunManager.PHASE_COMBAT and any(
        a.get("action") in PENDING_CHOICE_ACTIONS for a in actions
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
                    is_unknown=point_type in UNKNOWN_POINT_TYPES,
                    payload=dict(act),
                )
            )
        return cands

    if phase == RunManager.PHASE_CARD_REWARD:
        if is_potion_or_relic_reward(actions):
            return cands
        cands.extend(build_options(actions, _legal))
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
        if any(c.kind in {"potion", "relic"} for c in cands):
            return DECISION_POTION_OR_RELIC
        return DECISION_CARD_REWARD
    if phase == RunManager.PHASE_REST_SITE:
        return DECISION_REST_SITE
    if phase == RunManager.PHASE_EVENT:
        return DECISION_EVENT
    if phase == RunManager.PHASE_SHOP:
        return DECISION_SHOP
    if phase in {
        RunManager.PHASE_BOSS_RELIC,
        RunManager.PHASE_TREASURE,
    }:
        return DECISION_SIMILAR
    if cands:
        return DECISION_SIMILAR
    return DECISION_NONE


def _jev_phase_token(phase: str) -> str | None:
    if phase == RunManager.PHASE_MAP_CHOICE:
        return "map"
    if phase == RunManager.PHASE_REST_SITE:
        return "rest"
    if phase == RunManager.PHASE_CARD_REWARD:
        return "card"
    if phase == RunManager.PHASE_EVENT:
        return "event"
    return None


def _is_leave_option(cand: Candidate) -> bool:
    oid = str(cand.payload.get("option_id") or cand.key or "").strip().lower()
    label = str(cand.payload.get("label") or "").strip().lower()
    if oid in {"leave", "leave_event"}:
        return True
    if label == "leave":
        return True
    return cand.key.strip().lower() == "leave"


def _non_leave_count(cands: list[Candidate]) -> int:
    return sum(1 for c in cands if not _is_leave_option(c))


def _is_unknown_node(cand: Candidate) -> bool:
    if cand.is_unknown:
        return True
    return str(cand.payload.get("point_type", "")).upper() == "UNKNOWN"


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
        if _is_unknown_node(c):
            desc = f"{desc}. {UNKNOWN_MAP_CRITERION}"
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


def _fallback_legal_action(
    cands: list[Candidate],
    mask: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    if cands:
        return _legal_random_from(cands, rng)
    valid = np.flatnonzero(np.asarray(mask) == 1)
    return int(rng.choice(valid)) if valid.size else 0


def _hp_pressure_question() -> dict[str, Any]:
    return {
        "type": "score",
        "instructions": (
            "Score current HP pressure for an Act1 map fork. "
            f"Reference {CONTENT_MAP_REF}. Higher means rest is more urgent."
        ),
        "criteria": HP_PRESSURE_SCORE_CRITERIA,
    }


def _resolve_hp_pressure(
    pressure_ans: JevAnswer,
    mgr: RunManager,
) -> tuple[float, str]:
    hp = int(mgr.run_state.player.current_hp)
    max_hp = int(mgr.run_state.player.max_hp)
    if pressure_ans.status == "ok" and pressure_ans.score is not None:
        return float(pressure_ans.score), "jev_score"
    return local_hp_pressure(hp, max_hp), "local_fallback"


def _maybe_defer_unknown(
    cands: list[Candidate],
    chosen: Candidate,
    pick: JevAnswer,
    pressure: float,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer] | None:
    """Defer Unknown when HP is high-pressure and Choice is shy of 0.80."""
    if not _is_unknown_node(chosen):
        return None
    if pick.status != "ok":
        return None
    conf = pick.confidence
    if conf is None or conf >= UNKNOWN_DEFER_CONF:
        return None
    if pressure < HP_PRESSURE_REST:
        return None
    non_unknown = [c for c in cands if not _is_unknown_node(c)]
    if not non_unknown:
        return None
    action = _legal_random_from(non_unknown, rng)
    pick.status = "uncertain"
    pick.fallback_reason = UNKNOWN_DEFERRED_REASON
    logger.warning(
        "Jev unknown_deferred hp_pressure=%s conf=%s from=%s action=%s",
        pressure,
        conf,
        chosen.key,
        action,
    )
    return action, pick


def _augment_log(
    log: dict[str, Any],
    *,
    decision: str,
    cands: list[Candidate],
    action: int,
    choice_id: str | None = None,
    event_id: str | None = None,
    is_neow: bool | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    log["shadow_decision"] = decision
    log["legal_ids"] = [c.key for c in cands]
    executed = next((c.key for c in cands if c.run_action == action), None)
    log["executed_id"] = executed
    if log.get("shadow_confidence") is not None:
        log["jev_confidence"] = log["shadow_confidence"]
    if choice_id:
        log["jev_choice_id"] = choice_id
    if event_id is not None:
        log["event_id"] = event_id
    if is_neow is not None:
        log["is_neow"] = is_neow
    if phase:
        log["phase"] = phase
    return log


def choose_jev_noncombat(
    env: STS2RunEnv,
    mask: np.ndarray,
    rng: np.random.RandomState,
    adapter: JevClient,
    flags: JevPolicyFlags | None = None,
) -> tuple[int, dict[str, Any]]:
    """Pick a legal non-combat RunEnv action via Jev, or random on fallback.

    Errors are logged on the returned shadow fields (status=error) and the
    action falls back to legal random. The decision point is never skipped.
    Default flags keep EVENT off so MAP/REST/CARD tables stay comparable.
    """
    flags = flags or DEFAULT_JEV_FLAGS
    mgr = _mgr(env)
    actions = mgr.get_available_actions()
    if mgr.phase == RunManager.PHASE_CARD_REWARD and is_potion_or_relic_reward(actions):
        valid = np.flatnonzero(np.asarray(mask) == 1)
        action = int(rng.choice(valid)) if valid.size else 0
        log = JevAnswer(
            status="skipped",
            fallback_reason=POTION_OR_RELIC_REASON,
        ).as_log()
        log["shadow_decision"] = DECISION_POTION_OR_RELIC
        return action, log

    all_cands = collect_candidates(env, mask)
    cands = strip_illegal_invisible(all_cands)
    event_phase = mgr.phase == RunManager.PHASE_EVENT
    event_id = _event_id_from_mgr(mgr) if event_phase else ""
    is_neow = event_phase and is_neow_or_boon_screen(event_id, actions)
    if not cands and not event_phase:
        valid = np.flatnonzero(np.asarray(mask) == 1)
        action = int(rng.choice(valid)) if valid.size else 0
        logger.error("Jev status=error fallback=no_legal_visible_candidates; action=%s", action)
        log = JevAnswer(
            status="error",
            fallback_reason="no_legal_visible_candidates",
        ).as_log()
        return action, log

    decision = classify_decision(mgr.phase, cands)
    if is_neow:
        decision = DECISION_NEOW
    elif event_phase and not cands:
        decision = DECISION_EVENT
    state = _run_state_blob(mgr)
    state["decision"] = decision
    state["event_id"] = event_id
    state["is_neow"] = is_neow
    state["candidates"] = [
        {
            "key": c.key,
            "description": c.description,
            "upgraded": c.upgraded,
            "is_rest": c.is_rest,
            "is_unknown": _is_unknown_node(c),
        }
        for c in cands
    ]

    phase_token = _jev_phase_token(mgr.phase)
    allowed = flags.resolved_phases()
    skip_reason: str | None = None
    if decision == DECISION_SHOP or mgr.phase == RunManager.PHASE_SHOP:
        skip_reason = SHOP_RANDOM_REASON
    elif phase_token == "event" and not flags.allows_event():
        skip_reason = JEV_EVENT_OFF_REASON
    elif is_neow and not flags.allows_neow():
        skip_reason = JEV_NEOW_OFF_REASON
    elif phase_token == "event" and _non_leave_count(cands) < 2:
        skip_reason = NEOW_OPTIONS_EMPTY_REASON if is_neow else EVENT_OPTIONS_EMPTY_REASON
        logger.warning(
            "Jev %s non_leave=%s legal=%s; skipping Choice (not land-rate)",
            skip_reason,
            _non_leave_count(cands),
            [c.key for c in cands],
        )
    elif phase_token is None or phase_token not in allowed:
        skip_reason = NON_JEV_PHASE_REASON

    if skip_reason is not None:
        action = _fallback_legal_action(cands, mask, rng)
        log = JevAnswer(status="skipped", fallback_reason=skip_reason).as_log()
        log_phase = None
        if phase_token == "event":
            log_phase = "NEOW" if is_neow else "EVENT"
        return action, _augment_log(
            log,
            decision=decision,
            cands=cands,
            action=action,
            event_id=event_id or None,
            is_neow=is_neow if mgr.phase == RunManager.PHASE_EVENT else None,
            phase=log_phase,
        )

    try:
        action, answer = _decide(decision, cands, state, adapter, mgr, rng)
    except JevError as e:
        logger.error("Jev %s error; falling back to legal random: %s", decision, e)
        action = _legal_random_from(cands, rng)
        answer = JevAnswer(status="error", fallback_reason=str(e))
    log = answer.as_log()
    choice_id = None
    log_phase = None
    if decision == DECISION_EVENT:
        choice_id = CHOICE_EVENT
        log_phase = "EVENT"
    elif decision == DECISION_NEOW:
        choice_id = CHOICE_NEOW_BOON
        log_phase = "NEOW"
    log = _augment_log(
        log,
        decision=decision,
        cands=cands,
        action=action,
        choice_id=choice_id,
        event_id=event_id or None,
        is_neow=is_neow if mgr.phase == RunManager.PHASE_EVENT else None,
        phase=log_phase,
    )
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
    if decision == DECISION_CARD_REWARD:
        return _decide_card_reward(cands, state, adapter, rng)
    if decision in {DECISION_EVENT, DECISION_NEOW}:
        return _decide_event(decision, cands, state, adapter, rng)
    if decision == DECISION_MAP_FORK and any(_is_unknown_node(c) for c in cands):
        return _decide_map_fork_unknown(cands, state, adapter, mgr, rng)
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


def _decide_map_fork_unknown(
    cands: list[Candidate],
    state: dict[str, Any],
    adapter: JevClient,
    mgr: RunManager,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer]:
    questions = {
        "hp_pressure": _hp_pressure_question(),
        "pick": _choice_question(
            cands,
            _instructions_for(DECISION_MAP_FORK) + " " + UNKNOWN_MAP_CRITERION,
        ),
    }
    answers = adapter.system_one(state, questions)
    pressure, _src = _resolve_hp_pressure(
        answers.get("hp_pressure") or JevAnswer(status="error"), mgr
    )
    pick = apply_choice_confidence(answers.get("pick") or JevAnswer(status="error"))
    pick.score = pressure
    if pick.status != "ok":
        return _legal_random_from(cands, rng), pick
    chosen = _lookup(cands, pick.choice)
    if chosen is None:
        pick.status = "error"
        pick.fallback_reason = f"choice {pick.choice!r} not in legal candidates"
        return _legal_random_from(cands, rng), pick
    deferred = _maybe_defer_unknown(cands, chosen, pick, pressure, rng)
    if deferred is not None:
        return deferred
    return chosen.run_action, pick


def _decide_event(
    decision: str,
    cands: list[Candidate],
    state: dict[str, Any],
    adapter: JevClient,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer]:
    qname = CHOICE_NEOW_BOON if decision == DECISION_NEOW else CHOICE_EVENT
    instructions = (
        NEOW_BOON_INSTRUCTIONS if qname == CHOICE_NEOW_BOON else EVENT_CHOICE_INSTRUCTIONS
    )
    questions = {qname: _choice_question(cands, instructions)}
    answers = adapter.system_one(state, questions)
    pick = apply_choice_confidence(answers.get(qname) or JevAnswer(status="error"))
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
        "hp_pressure": _hp_pressure_question(),
        "pick": _choice_question(
            cands,
            (
                "Choose the next Act1 map node (rest vs continue). "
                f"See {CONTENT_MAP_REF}."
                + (
                    " " + UNKNOWN_MAP_CRITERION
                    if any(_is_unknown_node(c) for c in cands)
                    else ""
                )
            ),
        ),
    }
    answers = adapter.system_one(state, questions)
    pressure_ans = answers.get("hp_pressure") or JevAnswer(status="error")
    pick = apply_choice_confidence(answers.get("pick") or JevAnswer(status="error"))
    pressure, pressure_source = _resolve_hp_pressure(pressure_ans, mgr)

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
                deferred = _maybe_defer_unknown(
                    continue_cands, chosen, cont_pick, pressure, rng
                )
                if deferred is not None:
                    return deferred
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
    deferred = _maybe_defer_unknown(cands, chosen, pick, pressure, rng)
    if deferred is not None:
        return deferred
    return chosen.run_action, pick


def _is_skip_choice(chosen: Candidate | None, key: str | None) -> bool:
    if chosen is not None:
        return chosen.kind == "skip" or chosen.key == "card_skip"
    return key in {None, "skip", "card_skip"}


def _decide_card_reward(
    cands: list[Candidate],
    state: dict[str, Any],
    adapter: JevClient,
    rng: np.random.RandomState,
) -> tuple[int, JevAnswer]:
    questions = {
        "card_fit": {
            "type": "score",
            "instructions": (
                "Score how well the current Neow+early Act1 card offer fits "
                f"this deck. Reference {CONTENT_MAP_REF}. Higher means take "
                "the chosen card even if Choice confidence is shy of 0.65."
            ),
            "criteria": CARD_FIT_SCORE_CRITERIA,
        },
        "pick": _choice_question(cands, _instructions_for(DECISION_CARD_REWARD)),
    }
    answers = adapter.system_one(state, questions)
    pick = apply_choice_confidence(answers.get("pick") or JevAnswer(status="error"))
    fit_ans = answers.get("card_fit") or JevAnswer(status="error")
    card_fit = float(fit_ans.score) if fit_ans.status == "ok" and fit_ans.score is not None else None
    pick.card_fit = card_fit

    if pick.status == "ok":
        chosen = _lookup(cands, pick.choice)
        if chosen is None:
            pick.status = "error"
            pick.fallback_reason = f"choice {pick.choice!r} not in legal candidates"
            return _legal_random_from(cands, rng), pick
        return chosen.run_action, pick

    chosen = _lookup(cands, pick.choice)
    if (
        pick.status == "uncertain"
        and card_fit is not None
        and card_fit >= CARD_FIT_ASSIST_MIN
        and not _is_skip_choice(chosen, pick.choice)
        and chosen is not None
    ):
        pick.status = "ok"
        pick.fallback_reason = CARD_FIT_ASSIST_REASON
        return chosen.run_action, pick

    return _legal_random_from(cands, rng), pick


def _instructions_for(decision: str) -> str:
    base = f"Act1 Ironclad run. Criteria: {CONTENT_MAP_REF}. "
    if decision == DECISION_MAP_FORK:
        return base + "Choose the next map node among legal visible forks."
    if decision == DECISION_CARD_REWARD:
        return base + NEOW_EARLY_CARD_INSTRUCTIONS
    if decision == DECISION_REST_SITE:
        return base + "Choose a rest-site option (heal vs smith vs relic options)."
    if decision == DECISION_EVENT:
        return EVENT_CHOICE_INSTRUCTIONS
    if decision == DECISION_NEOW:
        return NEOW_BOON_INSTRUCTIONS
    return base + "Choose among legal visible options at this decision point."
