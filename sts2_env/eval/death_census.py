"""Act1 death-attribution census types and pure summarization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

RoomKind = Literal[
    "monster",
    "elite",
    "boss",
    "shop",
    "rest",
    "event",
    "treasure",
    "map",
    "other",
]
Outcome = Literal["won", "lost", "left"]
DeathKind = Literal[
    "combat_monster",
    "combat_elite",
    "combat_boss",
    "noncombat",
    "truncation",
    "cleared",
]

ALL_DEATH_KINDS: tuple[DeathKind, ...] = (
    "combat_monster",
    "combat_elite",
    "combat_boss",
    "noncombat",
    "truncation",
    "cleared",
)

COMBAT_ROOM_KINDS: tuple[Literal["monster", "elite", "boss"], ...] = (
    "monster",
    "elite",
    "boss",
)


@dataclass
class CombatEvent:
    floor: int
    act: int
    room_kind: Literal["monster", "elite", "boss"]
    encounter_id: str
    enemy_ids: list[str]
    hp_in: int
    hp_out: int
    max_hp: int
    gold_in: int
    gold_out: int
    outcome: Outcome
    steps: int


@dataclass
class ShopVisit:
    floor: int
    gold_in: int
    gold_out: int
    spent: int


@dataclass
class RunTrace:
    seed: int
    act1_clear: bool
    truncated: bool
    death_kind: DeathKind
    death_floor: int
    death_hp: int
    death_gold: int
    death_room_kind: RoomKind
    death_encounter_id: str
    death_enemy_ids: list[str]
    combats: list[CombatEvent] = field(default_factory=list)
    shop_visits: list[ShopVisit] = field(default_factory=list)
    rest_visits: int = 0
    event_visits: int = 0
    card_picks: int = 0
    card_skips: int = 0
    deck_size: int = 0
    upgraded_cards: int = 0
    relic_count: int = 0
    potion_count: int = 0


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals)) / len(vals)


def _shop_gold_spent(trace: RunTrace) -> int:
    return sum(v.spent for v in trace.shop_visits)


def trace_from_dict(data: dict[str, Any]) -> RunTrace:
    combats = [CombatEvent(**c) for c in data.get("combats", [])]
    shops = [ShopVisit(**s) for s in data.get("shop_visits", [])]
    return RunTrace(
        seed=int(data["seed"]),
        act1_clear=bool(data["act1_clear"]),
        truncated=bool(data["truncated"]),
        death_kind=data["death_kind"],
        death_floor=int(data["death_floor"]),
        death_hp=int(data["death_hp"]),
        death_gold=int(data["death_gold"]),
        death_room_kind=data["death_room_kind"],
        death_encounter_id=str(data.get("death_encounter_id", "")),
        death_enemy_ids=list(data.get("death_enemy_ids", [])),
        combats=combats,
        shop_visits=shops,
        rest_visits=int(data.get("rest_visits", 0)),
        event_visits=int(data.get("event_visits", 0)),
        card_picks=int(data.get("card_picks", 0)),
        card_skips=int(data.get("card_skips", 0)),
        deck_size=int(data.get("deck_size", 0)),
        upgraded_cards=int(data.get("upgraded_cards", 0)),
        relic_count=int(data.get("relic_count", 0)),
        potion_count=int(data.get("potion_count", 0)),
    )


def summarize_traces(traces: list[RunTrace]) -> dict[str, Any]:
    n = len(traces)
    if n == 0:
        counts = {k: 0 for k in ALL_DEATH_KINDS}
        rates = {k: 0.0 for k in ALL_DEATH_KINDS}
        return {
            "n": 0,
            "act1_clear_rate": 0.0,
            "trunc_rate": 0.0,
            "death_kind_counts": counts,
            "death_kind_rate": rates,
            "mean_death_floor": 0.0,
            "median_death_floor": 0.0,
            "mean_death_floor_by_death_kind": {},
            "median_death_floor_by_death_kind": {},
            "mean_death_gold": 0.0,
            "mean_gold_spent_in_shops_all_runs": 0.0,
            "mean_gold_spent_in_shops_deaths": 0.0,
            "combats_lost_by_room": {k: 0 for k in COMBAT_ROOM_KINDS},
            "mean_hp_lost_in_won_combats_by_room": {k: 0.0 for k in COMBAT_ROOM_KINDS},
            "mean_upgraded_cards_at_death": 0.0,
            "mean_upgraded_cards_at_clear": 0.0,
            "mean_deck_size_at_death": 0.0,
            "mean_deck_size_at_clear": 0.0,
            "mean_relic_count_at_death": 0.0,
            "mean_relic_count_at_clear": 0.0,
            "n_deaths_with_unspent_gold_ge_99": 0,
        }

    act1_clear_rate = sum(1 for t in traces if t.act1_clear) / n
    trunc_rate = sum(1 for t in traces if t.truncated) / n

    death_kind_counts: dict[DeathKind, int] = {k: 0 for k in ALL_DEATH_KINDS}
    for t in traces:
        death_kind_counts[t.death_kind] += 1
    death_kind_rate = {k: death_kind_counts[k] / n for k in ALL_DEATH_KINDS}

    deaths = [t for t in traces if t.death_kind != "cleared"]
    death_floors = [float(t.death_floor) for t in deaths]
    death_floor_mean = _mean(death_floors)
    death_floor_median = float(median(death_floors)) if death_floors else 0.0

    death_floor_mean_by_death_kind: dict[str, float] = {}
    death_floor_median_by_death_kind: dict[str, float] = {}
    for kind in ALL_DEATH_KINDS:
        if kind == "cleared":
            continue
        floors = [float(t.death_floor) for t in deaths if t.death_kind == kind]
        if not floors:
            continue
        death_floor_mean_by_death_kind[kind] = _mean(floors)
        death_floor_median_by_death_kind[kind] = float(median(floors))

    death_gold_mean = _mean([float(t.death_gold) for t in deaths])

    shop_spent_all = [_shop_gold_spent(t) for t in traces]
    shop_spent_deaths = [_shop_gold_spent(t) for t in deaths]
    gold_spent_in_shops_mean_all = _mean([float(x) for x in shop_spent_all])
    gold_spent_in_shops_mean_deaths = _mean([float(x) for x in shop_spent_deaths])

    combats_lost_by_room: dict[str, int] = {k: 0 for k in COMBAT_ROOM_KINDS}
    hp_lost_won: dict[str, list[float]] = {k: [] for k in COMBAT_ROOM_KINDS}
    for t in traces:
        for c in t.combats:
            if c.outcome == "lost":
                combats_lost_by_room[c.room_kind] += 1
            elif c.outcome == "won":
                hp_lost_won[c.room_kind].append(float(c.hp_in - c.hp_out))
    hp_lost_in_won_combats_mean_by_room = {
        k: _mean(v) for k, v in hp_lost_won.items()
    }

    clear_traces = [t for t in traces if t.death_kind == "cleared"]
    upgraded_cards_mean_at_death = _mean([float(t.upgraded_cards) for t in deaths])
    upgraded_cards_mean_at_clear = _mean([float(t.upgraded_cards) for t in clear_traces])
    deck_size_mean_at_death = _mean([float(t.deck_size) for t in deaths])
    deck_size_mean_at_clear = _mean([float(t.deck_size) for t in clear_traces])
    relic_count_mean_at_death = _mean([float(t.relic_count) for t in deaths])
    relic_count_mean_at_clear = _mean([float(t.relic_count) for t in clear_traces])

    n_deaths_with_unspent_gold_ge_99 = sum(
        1 for t in deaths if t.death_gold >= 99
    )

    return {
        "n": n,
        "act1_clear_rate": act1_clear_rate,
        "trunc_rate": trunc_rate,
        "death_kind_counts": death_kind_counts,
        "death_kind_rate": death_kind_rate,
        "mean_death_floor": death_floor_mean,
        "median_death_floor": death_floor_median,
        "mean_death_floor_by_death_kind": death_floor_mean_by_death_kind,
        "median_death_floor_by_death_kind": death_floor_median_by_death_kind,
        "mean_death_gold": death_gold_mean,
        "mean_gold_spent_in_shops_all_runs": gold_spent_in_shops_mean_all,
        "mean_gold_spent_in_shops_deaths": gold_spent_in_shops_mean_deaths,
        "combats_lost_by_room": combats_lost_by_room,
        "mean_hp_lost_in_won_combats_by_room": hp_lost_in_won_combats_mean_by_room,
        "mean_upgraded_cards_at_death": upgraded_cards_mean_at_death,
        "mean_upgraded_cards_at_clear": upgraded_cards_mean_at_clear,
        "mean_deck_size_at_death": deck_size_mean_at_death,
        "mean_deck_size_at_clear": deck_size_mean_at_clear,
        "mean_relic_count_at_death": relic_count_mean_at_death,
        "mean_relic_count_at_clear": relic_count_mean_at_clear,
        "n_deaths_with_unspent_gold_ge_99": n_deaths_with_unspent_gold_ge_99,
    }


def write_census(path: Path | str, traces: list[RunTrace], summary: dict[str, Any]) -> None:
    out = Path(path)
    payload = {
        "summary": summary,
        "traces": [asdict(t) for t in traces],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ALL_DEATH_KINDS",
    "CombatEvent",
    "DeathKind",
    "Outcome",
    "RoomKind",
    "RunTrace",
    "ShopVisit",
    "summarize_traces",
    "trace_from_dict",
    "write_census",
]
