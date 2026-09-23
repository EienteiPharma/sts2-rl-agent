"""Low-HP MAP fork routing helpers and constants (extracted from jev.py).

Owns MAP low-HP thresholds, safe/fight point predicates, v1 soft uncertain
filter, narrow soft-B elite/Boss avoidance, and opt-in v2 hard-select.
"""
from __future__ import annotations

from typing import Any

from sts2_env.core.enums import MapPointType
from sts2_env.map.generator import MapCoord

HP_PRESSURE_REST = 2.0
MAP_LOWHP_ON = True
MAP_LOWHP_PRESSURE = HP_PRESSURE_REST  # 2.0; same band as rest_or_continue prefer-rest
MAP_SAFE_POINT_TYPES = frozenset({"SHOP", "REST_SITE"})
MAP_FIGHT_POINT_TYPES = frozenset({"MONSTER", "ELITE", "BOSS"})
MAP_LOWHP_SAFE_REASON = "map_lowhp_safe"  # v1 fight-override; not hang-default
MAP_LOWHP_RANDOM_REASON = "map_lowhp_random"  # v1 uncertain filter (hang default on)
MAP_LOWHP_HARD_REASON = "map_lowhp_hard"  # v2 hard-select; --map-lowhp-hard, default off
MAP_LOWHP_HARD_ON = False
MAP_LOWHP_SOFT_B_REASON = "map_lowhp_soft_b"  # soft bias away from elite/boss under pressure
MAP_LOWHP_SOFT_B_ON = False

__all__ = [
    "MAP_FIGHT_POINT_TYPES",
    "MAP_LOWHP_HARD_ON",
    "MAP_LOWHP_HARD_REASON",
    "MAP_LOWHP_ON",
    "MAP_LOWHP_PRESSURE",
    "MAP_LOWHP_RANDOM_REASON",
    "MAP_LOWHP_SAFE_REASON",
    "MAP_LOWHP_SOFT_B_ON",
    "MAP_LOWHP_SOFT_B_REASON",
    "MAP_SAFE_POINT_TYPES",
    "cand_has_elite_or_boss_ahead",
    "is_map_fight_point",
    "is_map_safe_point",
    "map_lowhp_active",
    "map_lowhp_filter",
    "map_lowhp_filter_with_reason",
    "map_lowhp_hard_item",
    "map_lowhp_prefer",
    "map_lowhp_safe_items",
    "normalize_map_point_type",
]


def normalize_map_point_type(point_type: str | None) -> str:
    return str(point_type or "").strip().upper().replace("-", "_")


def is_map_safe_point(point_type: str | None) -> bool:
    """Shop or rest map nodes — legal recoveries when HP is thin."""
    name = normalize_map_point_type(point_type)
    if name in MAP_SAFE_POINT_TYPES:
        return True
    if name in {"RESTSITE", "REST"}:
        return True
    if name in {"MERCHANT", "STORE"}:
        return True
    return False


def is_map_fight_point(point_type: str | None) -> bool:
    return normalize_map_point_type(point_type) in MAP_FIGHT_POINT_TYPES


def map_lowhp_active(
    hp_pressure: float | None,
    *,
    enabled: bool = MAP_LOWHP_ON,
) -> bool:
    return bool(enabled) and hp_pressure is not None and float(hp_pressure) >= MAP_LOWHP_PRESSURE


def map_lowhp_safe_items(items, point_type_of):
    return [item for item in items if is_map_safe_point(point_type_of(item))]


def map_lowhp_filter(items, point_type_of, hp_pressure, *, enabled: bool = MAP_LOWHP_ON):
    """Safe shop/rest subset when the constraint fires.

    Returns None when the filter does not apply (caller keeps the full pool),
    including the documented case where only monster/elite remain.
    """
    if not map_lowhp_active(hp_pressure, enabled=enabled):
        return None
    safe = map_lowhp_safe_items(items, point_type_of)
    return safe or None


def cand_has_elite_or_boss_ahead(item, point_type_of, act_map: Any = None) -> bool:
    """True if item is or directly leads into an ELITE or BOSS."""
    pt = normalize_map_point_type(point_type_of(item))
    if pt in {"ELITE", "BOSS"}:
        return True
    if act_map is None:
        return False
    coord = None
    if hasattr(item, "payload") and isinstance(item.payload, dict):
        coord = item.payload.get("coord")
    elif hasattr(item, "meta") and isinstance(item.meta, dict):
        coord = item.meta.get("coord")
    if not coord or len(coord) < 2:
        return False

    try:
        mp = act_map.get_point(MapCoord(coord[0], coord[1]))
        if mp is None:
            return False
        if getattr(mp, "point_type", None) in (MapPointType.ELITE, MapPointType.BOSS):
            return True
        frontier = list(getattr(mp, "children", []))
        for _ in range(2):
            next_frontier = []
            for child in frontier:
                if getattr(child, "point_type", None) in (MapPointType.ELITE, MapPointType.BOSS):
                    return True
                next_frontier.extend(getattr(child, "children", []))
            frontier = next_frontier
    except Exception:
        pass
    return False


def map_lowhp_filter_with_reason(
    items,
    point_type_of,
    hp_pressure,
    *,
    enabled: bool = MAP_LOWHP_ON,
    soft_b: bool = MAP_LOWHP_SOFT_B_ON,
    act_map: Any = None,
) -> tuple[Any | None, str | None]:
    """Soft filter on uncertain/error map forks under hp_pressure >= 2.0.

    When soft_b is enabled and an elite/Boss is ahead on any fork option,
    and a safe node (SHOP/REST_SITE) is present, soft-prefers the safe node(s)
    to avoid heading into elite/boss under dangerous HP pressure.
    Falls back to v1 safe-node filter if enabled.
    Returns (filtered_pool, reason) or (None, None).
    """
    if not map_lowhp_active(hp_pressure, enabled=enabled or soft_b):
        return None, None

    safe = map_lowhp_safe_items(items, point_type_of)
    if not safe:
        # If only fight nodes remain, keep full-pool random
        return None, None

    danger = [it for it in items if cand_has_elite_or_boss_ahead(it, point_type_of, act_map)]
    if soft_b and danger:
        safe_non_danger = [it for it in safe if it not in danger]
        pool = safe_non_danger or safe
        if pool:
            return pool, MAP_LOWHP_SOFT_B_REASON

    if enabled and safe:
        return safe, MAP_LOWHP_RANDOM_REASON

    return None, None


def map_lowhp_hard_item(items, point_type_of, hp_pressure, *, enabled: bool = MAP_LOWHP_HARD_ON):
    """Rest-then-shop pick when the v2 hard-select fires; else None.

    None means the caller keeps Choice / full-pool random (only-fight forks,
    pressure below threshold, ``--map-lowhp-hard off``, or hang default).
    Hang default is **off** (``MAP_LOWHP_HARD_ON = False``).
    """
    if not map_lowhp_active(hp_pressure, enabled=enabled):
        return None
    safe = map_lowhp_safe_items(items, point_type_of)
    if not safe:
        return None
    return map_lowhp_prefer(safe, point_type_of)


def map_lowhp_prefer(items, point_type_of):
    """Deterministic safe pick: rest before shop."""
    if not items:
        raise ValueError("map_lowhp_prefer requires a non-empty safe pool")
    rests = [
        item
        for item in items
        if normalize_map_point_type(point_type_of(item)) in {"REST_SITE", "RESTSITE", "REST"}
    ]
    if rests:
        return rests[0]
    shops = [
        item
        for item in items
        if normalize_map_point_type(point_type_of(item)) in {"SHOP", "MERCHANT", "STORE"}
    ]
    if shops:
        return shops[0]
    return items[0]
