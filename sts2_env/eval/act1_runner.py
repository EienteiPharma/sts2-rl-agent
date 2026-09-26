"""Act1 RunEnv eval runner: model load, action choice, and episode loop.

Policy / Jev / map_lowhp behavior is unchanged; this module only owns the
eval wiring previously in ``scripts/eval_act1_runenv.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sts2_env.eval.act1_suite import (
    JEV_SHADOW_SKIPPED,
    JEV_SHADOW_STUB,
    SEED_COUNT,
)
from sts2_env.eval.combat_jev import (
    CombatJevTelemetry,
    choose_combat_step,
    hung_ppo_local,
)
from sts2_env.eval.jev_client import build_jev_adapter
from sts2_env.eval.jev_config import (
    DEFAULT_JEV_FLAGS,
    JevPolicyFlags,
    resolve_jev_flags,
)
from sts2_env.eval.jev_policy import choose_jev_noncombat
from sts2_env.eval.jev_types import (
    EVENT_OPTIONS_EMPTY_REASON,
    EVENT_SAFE_FALLBACK_REASON,
    JEV_EVENT_OFF_REASON,
    POTION_OR_RELIC_REASON,
    POTION_OR_RELIC_SAFE_REASON,
    JevAnswer,
)
from sts2_env.eval.map_lowhp import (
    MAP_LOWHP_HARD_REASON,
    MAP_LOWHP_SOFT_B_REASON,
)
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.run_env import (
    RUN_OBS_SIZE,
    STS2RunEnv,
    _COMBAT_SIZE,
    _COMBAT_START,
)
from dataclasses import asdict

from sts2_env.core.enums import RoomType
from sts2_env.eval.death_census import (
    CombatEvent,
    DeathKind,
    Outcome,
    RoomKind,
    RunTrace,
    ShopVisit,
)
from sts2_env.run.rooms import CombatRoom
from sts2_env.run.run_manager import RunManager


def model_obs_dim(model: Any) -> int:
    """Read a loaded SB3 model's observation width."""
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def require_obs_dim(model: Any, expected: int, what: str) -> int:
    """Refuse a zip whose observation width does not match *expected*."""
    obs_dim = model_obs_dim(model)
    if obs_dim != expected:
        raise SystemExit(
            f"{what} obs_dim={obs_dim} != expected={expected}; "
            "combat-suite zips are not valid for --policy model (use "
            "--policy hierarchical --combat-model), and RunEnv zips are not "
            "valid as --combat-model"
        )
    return obs_dim


def load_maskable_ppo(path: str) -> Any:
    zip_path = Path(path)
    if not zip_path.is_file():
        raise SystemExit(f"model zip not found: {path}")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise SystemExit(
            "sb3-contrib is required to load MaskablePPO zips. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    return MaskablePPO.load(str(zip_path), device="cpu")


def _run_manager(env: STS2RunEnv) -> RunManager:
    mgr = getattr(env, "_mgr", None)
    if mgr is None:
        raise RuntimeError("STS2RunEnv has no run manager; call reset() first")
    return mgr


def _selected_combat_owner(mgr: RunManager, combat):
    actions = mgr.get_available_actions()
    selected_action = next(
        (a for a in actions if a.get("action") == "select_player" and a.get("selected")),
        None,
    )
    selected_owner = combat.primary_player
    if selected_action is not None:
        for state in combat.combat_player_states:
            if state.player_state.player_id == selected_action.get("player_id"):
                selected_owner = state.creature
                break
    return selected_owner


def jev_shadow_fields(phase: str, *, jev_enabled: bool = False) -> dict[str, Any]:
    """Shadow log for steps that did not run a live Jev decision."""
    if phase == RunManager.PHASE_COMBAT:
        return JevAnswer(status=JEV_SHADOW_SKIPPED).as_log()
    if jev_enabled:
        return JevAnswer(status=JEV_SHADOW_STUB, fallback_reason="non_decision").as_log()
    return JevAnswer(status=JEV_SHADOW_STUB, fallback_reason="jev_off").as_log()


def _combat_room_kind(mgr: RunManager) -> Literal["monster", "elite", "boss"]:
    room = mgr.current_room
    if isinstance(room, CombatRoom):
        if room.is_boss or room.room_type == RoomType.BOSS:
            return "boss"
        if room.is_elite or room.room_type == RoomType.ELITE:
            return "elite"
        return "monster"
    rt = mgr._current_room_type
    if rt == RoomType.BOSS:
        return "boss"
    if rt == RoomType.ELITE:
        return "elite"
    return "monster"


def _room_type_to_room_kind(rt: RoomType | None) -> RoomKind:
    if rt is None:
        return "other"
    mapping: dict[RoomType, RoomKind] = {
        RoomType.MONSTER: "monster",
        RoomType.ELITE: "elite",
        RoomType.BOSS: "boss",
        RoomType.SHOP: "shop",
        RoomType.REST_SITE: "rest",
        RoomType.EVENT: "event",
        RoomType.TREASURE: "treasure",
    }
    return mapping.get(rt, "other")


def _phase_to_room_kind(phase: str) -> RoomKind:
    if phase == RunManager.PHASE_MAP_CHOICE:
        return "map"
    if phase == RunManager.PHASE_SHOP:
        return "shop"
    if phase == RunManager.PHASE_REST_SITE:
        return "rest"
    if phase == RunManager.PHASE_EVENT:
        return "event"
    if phase == RunManager.PHASE_TREASURE:
        return "treasure"
    if phase == RunManager.PHASE_COMBAT:
        return "monster"
    return "other"


def _combat_enemy_ids(mgr: RunManager) -> list[str]:
    combat = mgr.get_combat_state()
    if combat is None:
        return []
    ids: list[str] = []
    for enemy in combat.enemies:
        mid = getattr(enemy, "monster_id", None)
        ids.append(str(mid) if mid is not None else "")
    return ids


def _combat_encounter_id(mgr: RunManager) -> str:
    room = mgr.current_room
    if isinstance(room, CombatRoom):
        return str(room.encounter_id or "")
    return ""


def _count_upgraded_cards(deck: list[Any]) -> int:
    n = 0
    for card in deck:
        if getattr(card, "upgraded", False):
            n += 1
    return n


def _legal_random(mask: np.ndarray, rng: np.random.RandomState) -> int:
    valid = np.flatnonzero(np.asarray(mask) == 1)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def choose_hierarchical_action(
    env: STS2RunEnv,
    obs: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    combat_model: Any,
    *,
    received_obs_widths: list[int] | None = None,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
    combat_policy: str = "ppo",
    combat_jev_adapter: Any = None,
    combat_jev_telemetry: CombatJevTelemetry | None = None,
) -> tuple[int, dict[str, Any]]:
    """Pick a RunEnv action for hierarchical policy.

    Combat default (``combat_policy=ppo``): encode *combat* obs (never ``obs``
    / RunEnv) and predict with hung bh_v1, then map the combat action index
    into the RunEnv combat slice. ``combat_policy=jev`` is experimental
    (failed HOLD bypass; not hang / not next mainline): TypeSafe Choice on
    the legal shortlist, fail-open to the same hung zip.
    Non-combat: legal random when ``jev_enabled`` is false; Jev Choice/Score
    when true (errors fall back to legal random). Ordinary EVENT is off unless
    ``jev_flags.allows_event()``. Detected Neow uses Jev only when
    ``jev_flags.allows_neow()`` is true (hang default: random boon).
    """
    mgr = _run_manager(env)
    phase = mgr.phase
    shadow = jev_shadow_fields(phase, jev_enabled=jev_enabled)

    if phase != RunManager.PHASE_COMBAT:
        if jev_enabled:
            adapter = jev_adapter or build_jev_adapter(enabled=True)
            return choose_jev_noncombat(
                env, mask, rng, adapter, flags=jev_flags or DEFAULT_JEV_FLAGS
            )
        return _legal_random(mask, rng), shadow

    combat = mgr.get_combat_state()
    if combat is None:
        return _legal_random(mask, rng), shadow

    combat_obs = encode_observation(combat)
    if combat_obs.shape[-1] != OBS_SIZE:
        raise RuntimeError(
            f"combat obs width {combat_obs.shape[-1]} != OBS_SIZE={OBS_SIZE}"
        )
    if np.asarray(obs).shape[-1] == combat_obs.shape[-1]:
        raise RuntimeError(
            "RunEnv obs width equals combat OBS_SIZE; hierarchical eval "
            "refuses to treat run observations as combat observations"
        )
    if received_obs_widths is not None:
        received_obs_widths.append(int(combat_obs.shape[-1]))

    owner = _selected_combat_owner(mgr, combat)
    combat_mask = get_action_mask(combat, owner=owner)
    if combat_policy == "jev-turn":
        from sts2_env.eval.combat_turn_plan import choose_combat_turn_plan_action

        adapter = combat_jev_adapter or jev_adapter or build_jev_adapter(enabled=True)
        local, shadow = choose_combat_turn_plan_action(
            combat,
            combat_mask,
            rng,
            combat_model,
            adapter=adapter,
            combat_obs=combat_obs,
            env=env,
            owner=owner,
            telemetry=combat_jev_telemetry,
        )
        local = max(0, min(int(local), _COMBAT_SIZE - 1))
        return _COMBAT_START + local, shadow
    if combat_policy == "jev":
        adapter = combat_jev_adapter or jev_adapter or build_jev_adapter(enabled=True)
        local, shadow = choose_combat_step(
            combat,
            combat_mask,
            rng,
            combat_model,
            adapter=adapter,
            combat_obs=combat_obs,
            telemetry=combat_jev_telemetry,
            owner=owner,
        )
        local = max(0, min(int(local), _COMBAT_SIZE - 1))
        return _COMBAT_START + local, shadow
    local = hung_ppo_local(combat_model, combat_obs, combat_mask)
    if local is None:
        local = _legal_random(combat_mask, rng)
    local = max(0, min(int(local), _COMBAT_SIZE - 1))
    return _COMBAT_START + local, shadow


def choose_action(
    policy: str,
    env: STS2RunEnv,
    obs: np.ndarray,
    info: dict[str, Any],
    mask: np.ndarray,
    rng: np.random.RandomState,
    model: Any,
    combat_model: Any,
    *,
    received_obs_widths: list[int] | None = None,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
    combat_policy: str = "ppo",
    combat_jev_adapter: Any = None,
    combat_jev_telemetry: CombatJevTelemetry | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return ``(action, shadow_fields)`` for the current step."""
    if policy == "random":
        return _legal_random(mask, rng), jev_shadow_fields(info.get("phase", ""))
    if policy == "model":
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action), jev_shadow_fields(info.get("phase", ""))
    if policy == "hierarchical":
        return choose_hierarchical_action(
            env,
            obs,
            mask,
            rng,
            combat_model,
            received_obs_widths=received_obs_widths,
            jev_enabled=jev_enabled,
            jev_adapter=jev_adapter,
            jev_flags=jev_flags,
            combat_policy=combat_policy,
            combat_jev_adapter=combat_jev_adapter,
            combat_jev_telemetry=combat_jev_telemetry,
        )
    raise SystemExit(f"unknown policy: {policy}")


def _run_episode(
    env: STS2RunEnv,
    policy: str,
    model: Any,
    combat_model: Any,
    seed: int,
    rng: np.random.RandomState,
    *,
    jev_enabled: bool = False,
    jev_adapter: Any = None,
    jev_flags: JevPolicyFlags | None = None,
    start_with_neow: bool = False,
    combat_policy: str = "ppo",
    combat_jev_adapter: Any = None,
    combat_jev_telemetry: CombatJevTelemetry | None = None,
    death_census: bool = False,
) -> dict:
    obs, info = env.reset(seed=seed, options={"start_with_neow": start_with_neow})
    done = False
    ep_rew = 0.0
    steps = 0
    combat_steps = 0
    noncombat_steps = 0
    max_act = int(info.get("act", 0))
    reward = 0.0
    terminated = False
    truncated = False
    last_shadow = jev_shadow_fields(info.get("phase", ""), jev_enabled=jev_enabled)
    map_lowhp_hard_n = 0
    map_lowhp_soft_b_n = 0
    event_jev_used_n = 0
    event_safe_fallback_n = 0
    event_low_conf_random_n = 0
    event_options_empty_n = 0
    event_off_random_n = 0
    potion_or_relic_safe_n = 0
    potion_or_relic_random_n = 0
    tel_mark = combat_jev_telemetry.mark() if combat_jev_telemetry is not None else (0, 0)
    census_combats: list[CombatEvent] = []
    census_shop_visits: list[ShopVisit] = []
    census_rest_visits = 0
    census_event_visits = 0
    census_card_picks = 0
    census_card_skips = 0
    census_open_combat: dict[str, Any] | None = None
    census_shop_open: dict[str, int] | None = None
    census_card_deck_len: int | None = None
    census_last_lost_combat: CombatEvent | None = None
    census_last_noncombat_phase: str | None = None

    def _census_floor_act() -> tuple[int, int]:
        return int(info.get("floor", 0)), int(info.get("act", 0))

    def _census_start_combat(mgr: RunManager) -> None:
        nonlocal census_open_combat
        player = mgr.run_state.player
        floor, act = _census_floor_act()
        census_open_combat = {
            "floor": floor,
            "act": act,
            "room_kind": _combat_room_kind(mgr),
            "encounter_id": _combat_encounter_id(mgr),
            "enemy_ids": _combat_enemy_ids(mgr),
            "hp_in": int(player.current_hp),
            "gold_in": int(player.gold),
            "max_hp": int(player.max_hp),
            "steps": 0,
        }

    def _census_end_combat(mgr: RunManager) -> None:
        nonlocal census_open_combat, census_last_lost_combat
        if census_open_combat is None:
            return
        player = mgr.run_state.player
        hp_out = int(player.current_hp)
        gold_out = int(player.gold)
        still_combat = mgr.phase == RunManager.PHASE_COMBAT
        outcome: Outcome = "left"
        if hp_out <= 0:
            outcome = "lost"
        elif not still_combat:
            outcome = "won"
        evt = CombatEvent(
            floor=int(census_open_combat["floor"]),
            act=int(census_open_combat["act"]),
            room_kind=census_open_combat["room_kind"],
            encounter_id=str(census_open_combat["encounter_id"]),
            enemy_ids=list(census_open_combat["enemy_ids"]),
            hp_in=int(census_open_combat["hp_in"]),
            hp_out=hp_out,
            max_hp=int(census_open_combat["max_hp"]),
            gold_in=int(census_open_combat["gold_in"]),
            gold_out=gold_out,
            outcome=outcome,
            steps=int(census_open_combat["steps"]),
        )
        census_combats.append(evt)
        if outcome == "lost":
            census_last_lost_combat = evt
        census_open_combat = None

    def _census_on_phase_enter(phase: str, mgr: RunManager) -> None:
        nonlocal census_rest_visits, census_event_visits, census_shop_open, census_card_deck_len
        if phase == RunManager.PHASE_REST_SITE:
            census_rest_visits += 1
        elif phase == RunManager.PHASE_EVENT:
            census_event_visits += 1
        elif phase == RunManager.PHASE_SHOP:
            census_shop_open = {
                "floor": int(info.get("floor", 0)),
                "gold_in": int(mgr.run_state.player.gold),
            }
        elif phase == RunManager.PHASE_CARD_REWARD:
            census_card_deck_len = len(mgr.run_state.player.deck)

    def _census_on_phase_leave(phase: str, mgr: RunManager) -> None:
        nonlocal census_shop_open, census_card_picks, census_card_skips, census_card_deck_len
        if phase == RunManager.PHASE_SHOP and census_shop_open is not None:
            gold_out = int(mgr.run_state.player.gold)
            gold_in = int(census_shop_open["gold_in"])
            census_shop_visits.append(
                ShopVisit(
                    floor=int(census_shop_open["floor"]),
                    gold_in=gold_in,
                    gold_out=gold_out,
                    spent=max(0, gold_in - gold_out),
                )
            )
            census_shop_open = None
        elif phase == RunManager.PHASE_CARD_REWARD and census_card_deck_len is not None:
            deck_len = len(mgr.run_state.player.deck)
            if deck_len > census_card_deck_len:
                census_card_picks += 1
            else:
                census_card_skips += 1
            census_card_deck_len = None

    if death_census:
        mgr0 = _run_manager(env)
        phase0 = str(info.get("phase", mgr0.phase))
        if phase0 == RunManager.PHASE_COMBAT:
            _census_start_combat(mgr0)
        else:
            _census_on_phase_enter(phase0, mgr0)
            census_last_noncombat_phase = phase0

    while not done:
        phase_before = str(info.get("phase", _run_manager(env).phase))
        if death_census:
            mgr_before = _run_manager(env)
            if phase_before == RunManager.PHASE_COMBAT and census_open_combat is None:
                _census_start_combat(mgr_before)
            if phase_before == RunManager.PHASE_COMBAT and census_open_combat is not None:
                census_open_combat["steps"] = int(census_open_combat["steps"]) + 1

        mask = info.get("action_mask")
        if mask is None:
            mask = env.action_masks()
        mask = np.asarray(mask)
        action, last_shadow = choose_action(
            policy,
            env,
            obs,
            info,
            mask,
            rng,
            model,
            combat_model,
            jev_enabled=jev_enabled,
            jev_adapter=jev_adapter,
            jev_flags=jev_flags,
            combat_policy=combat_policy,
            combat_jev_adapter=combat_jev_adapter,
            combat_jev_telemetry=combat_jev_telemetry,
        )
        reason = last_shadow.get("shadow_fallback_reason")
        dec = last_shadow.get("shadow_decision")
        status = last_shadow.get("shadow_status")
        if (
            reason == MAP_LOWHP_HARD_REASON
            or last_shadow.get("map_lowhp_hard")
        ):
            map_lowhp_hard_n += 1
            print(
                f"map_lowhp_hard seed={seed} n={map_lowhp_hard_n} "
                f"executed={last_shadow.get('executed_id')} "
                f"jev_choice={last_shadow.get('shadow_suggestion')}",
                file=sys.stderr,
                flush=True,
            )
        if (
            reason == MAP_LOWHP_SOFT_B_REASON
            or last_shadow.get("map_lowhp_soft_b")
        ):
            map_lowhp_soft_b_n += 1
            print(
                f"map_lowhp_soft_b seed={seed} n={map_lowhp_soft_b_n} "
                f"executed={last_shadow.get('executed_id')} "
                f"jev_choice={last_shadow.get('shadow_suggestion')}",
                file=sys.stderr,
                flush=True,
            )
        if dec == "event_choice" or last_shadow.get("phase") == "EVENT":
            if status == "ok" and not reason:
                event_jev_used_n += 1
            elif reason == EVENT_SAFE_FALLBACK_REASON or last_shadow.get("event_safe_fallback"):
                event_safe_fallback_n += 1
            elif reason == "low_confidence_random":
                event_low_conf_random_n += 1
            elif reason == EVENT_OPTIONS_EMPTY_REASON:
                event_options_empty_n += 1
            elif reason == JEV_EVENT_OFF_REASON:
                event_off_random_n += 1
        if dec == "potion_or_relic_reward":
            if reason == POTION_OR_RELIC_SAFE_REASON or last_shadow.get("potion_or_relic_safe_fallback"):
                potion_or_relic_safe_n += 1
            elif reason == POTION_OR_RELIC_REASON:
                potion_or_relic_random_n += 1
        if info.get("phase") == RunManager.PHASE_COMBAT:
            combat_steps += 1
        else:
            noncombat_steps += 1
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += float(reward)
        steps += 1
        max_act = max(max_act, int(info.get("act", 0)))
        done = terminated or truncated
        if death_census:
            mgr_after = _run_manager(env)
            phase_after = str(info.get("phase", mgr_after.phase))
            if phase_before == RunManager.PHASE_COMBAT and (
                phase_after != RunManager.PHASE_COMBAT or done
            ):
                _census_end_combat(mgr_after)
            if phase_before != phase_after:
                _census_on_phase_leave(phase_before, mgr_after)
                _census_on_phase_enter(phase_after, mgr_after)
                if phase_after != RunManager.PHASE_COMBAT:
                    census_last_noncombat_phase = phase_after
    act1_clear = bool(max_act >= 1)
    row: dict[str, Any] = {
        "seed": seed,
        "act1_clear": act1_clear,
        "full_run_win": bool(terminated and reward > 0),
        "truncated": bool(truncated),
        "max_act": max_act,
        "floor": int(info.get("floor", 0)),
        "hp": int(info.get("hp", 0)),
        "max_hp": int(info.get("max_hp", 0)),
        "gold": int(info.get("gold", 0)),
        "steps": steps,
        "combat_steps": combat_steps,
        "noncombat_steps": noncombat_steps,
        "reward": ep_rew,
        "shadow_suggestion": last_shadow.get("shadow_suggestion"),
        "shadow_status": last_shadow.get("shadow_status"),
        "shadow_confidence": last_shadow.get("shadow_confidence"),
        "shadow_hp_pressure": last_shadow.get("shadow_hp_pressure"),
        "shadow_fallback_reason": last_shadow.get("shadow_fallback_reason"),
        "map_lowhp_hard_n": map_lowhp_hard_n,
        "map_lowhp_soft_b_n": map_lowhp_soft_b_n,
        "event_jev_used_n": event_jev_used_n,
        "event_safe_fallback_n": event_safe_fallback_n,
        "event_low_conf_random_n": event_low_conf_random_n,
        "event_options_empty_n": event_options_empty_n,
        "event_off_random_n": event_off_random_n,
        "potion_or_relic_safe_n": potion_or_relic_safe_n,
        "potion_or_relic_random_n": potion_or_relic_random_n,
        "jev_card_fit": last_shadow.get("jev_card_fit"),
        **(
            combat_jev_telemetry.episode_fields(tel_mark)
            if combat_jev_telemetry is not None
            else {"combat_jev_calls": 0, "combat_jev_fail_open": 0}
        ),
    }
    if death_census:
        mgr_end = _run_manager(env)
        if census_open_combat is not None:
            _census_end_combat(mgr_end)
        if census_shop_open is not None:
            _census_on_phase_leave(RunManager.PHASE_SHOP, mgr_end)
        if census_card_deck_len is not None:
            _census_on_phase_leave(RunManager.PHASE_CARD_REWARD, mgr_end)
        player = mgr_end.run_state.player
        trunc_flag = bool(truncated)
        death_kind: DeathKind
        if act1_clear:
            death_kind = "cleared"
        elif trunc_flag:
            death_kind = "truncation"
        elif census_last_lost_combat is not None:
            rk = census_last_lost_combat.room_kind
            death_kind = f"combat_{rk}"  # type: ignore[assignment]
        else:
            death_kind = "noncombat"
        death_room_kind: RoomKind = "other"
        death_encounter_id = ""
        death_enemy_ids: list[str] = []
        if census_last_lost_combat is not None:
            death_room_kind = census_last_lost_combat.room_kind
            death_encounter_id = census_last_lost_combat.encounter_id
            death_enemy_ids = list(census_last_lost_combat.enemy_ids)
        elif census_last_noncombat_phase is not None:
            if census_last_noncombat_phase == RunManager.PHASE_MAP_CHOICE:
                death_room_kind = "map"
            else:
                death_room_kind = _phase_to_room_kind(census_last_noncombat_phase)
                rt = mgr_end._current_room_type
                if rt is not None and death_room_kind == "other":
                    death_room_kind = _room_type_to_room_kind(rt)
        trace = RunTrace(
            seed=seed,
            act1_clear=act1_clear,
            truncated=trunc_flag,
            death_kind=death_kind,
            death_floor=int(info.get("floor", 0)),
            death_hp=int(info.get("hp", 0)),
            death_gold=int(info.get("gold", 0)),
            death_room_kind=death_room_kind,
            death_encounter_id=death_encounter_id,
            death_enemy_ids=death_enemy_ids,
            combats=census_combats,
            shop_visits=census_shop_visits,
            rest_visits=census_rest_visits,
            event_visits=census_event_visits,
            card_picks=census_card_picks,
            card_skips=census_card_skips,
            deck_size=len(player.deck),
            upgraded_cards=_count_upgraded_cards(player.deck),
            relic_count=len(player.relics),
            potion_count=len(player.held_potions()),
        )
        row["death_census"] = asdict(trace)
    return row


def validate_policy_args(args: argparse.Namespace) -> None:
    if getattr(args, "strategic", None) == "jev":
        args.jev = "on"
    if args.policy == "model":
        if not args.model:
            raise SystemExit("--model required for --policy model")
        if args.combat_model:
            raise SystemExit("--combat-model is only valid with --policy hierarchical")
    elif args.policy == "hierarchical":
        if args.model and args.combat_model and args.model != args.combat_model:
            raise SystemExit(
                "hierarchical: --model and --combat-model are the same combat "
                "zip; pass one path (Surplus: --model)"
            )
        zip_path = args.combat_model or args.model
        if not zip_path:
            raise SystemExit(
                "--model (hung combat zip, obs_v1=181) required for "
                "--policy hierarchical; --combat-model is an alias"
            )
        args.combat_model = zip_path
    elif args.combat_model or args.model:
        raise SystemExit("--model/--combat-model require --policy model or hierarchical")
    if args.jev == "on" and args.policy != "hierarchical":
        raise SystemExit("--jev on is only valid with --policy hierarchical")
    if args.policy != "hierarchical":
        if args.jev_event == "on":
            raise SystemExit("--jev-event on is only valid with --policy hierarchical")
        if args.jev_neow == "on":
            raise SystemExit("--jev-neow on is only valid with --policy hierarchical")
        cp = str(getattr(args, "combat_policy", "ppo") or "ppo")
        if cp in ("jev", "jev-turn"):
            raise SystemExit(
                f"--combat-policy {cp} is only valid with --policy hierarchical"
            )
    if int(getattr(args, "n", SEED_COUNT)) < 1:
        raise SystemExit("--n must be >= 1")
    args.jev_flags = resolve_jev_flags(
        jev_event=args.jev_event,
        jev_phases=args.jev_phases,
        jev_neow=args.jev_neow,
        map_lowhp=getattr(args, "map_lowhp", "on"),
        map_lowhp_hard=getattr(args, "map_lowhp_hard", "off"),
        map_lowhp_soft_b=getattr(args, "map_lowhp_soft_b", "off"),
    )


def load_policy_models(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load and type-check zips.

    ``--policy model`` requires RunEnv obs (201); combat zips are rejected.
    ``--policy hierarchical`` loads the hung combat zip (obs_v1=181) from
    ``--model`` or alias ``--combat-model``. Never feed RunEnv obs to it.
    """
    model = None
    combat_model = None
    if args.policy == "model":
        model = load_maskable_ppo(args.model)
        require_obs_dim(model, RUN_OBS_SIZE, "RunEnv --model")
    elif args.policy == "hierarchical":
        combat_path = args.combat_model or args.model
        combat_model = load_maskable_ppo(combat_path)
        require_obs_dim(combat_model, OBS_SIZE, "hierarchical combat zip (--model)")
    return model, combat_model


__all__ = [
    "_legal_random",
    "_run_episode",
    "_run_manager",
    "_selected_combat_owner",
    "choose_action",
    "choose_hierarchical_action",
    "jev_shadow_fields",
    "load_maskable_ppo",
    "load_policy_models",
    "model_obs_dim",
    "require_obs_dim",
    "validate_policy_args",
]
