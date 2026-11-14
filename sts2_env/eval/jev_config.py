"""Policy configuration and switch resolution for non-combat Jev evaluation.

Owns JevPolicyFlags, DEFAULT_JEV_FLAGS, phase parsing, and flag resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

from sts2_env.eval.jev_types import DEFAULT_JEV_PHASES, JEV_PHASE_TOKENS
from sts2_env.eval.map_lowhp import (
    MAP_LOWHP_HARD_ON,
    MAP_LOWHP_ON,
    MAP_LOWHP_SOFT_B_ON,
)


@dataclass(frozen=True)
class JevPolicyFlags:
    """Which non-combat phases call Jev. Default preserves MAP/REST/CARD tables.

    EVENT is off unless ``event`` / ``--jev-event on``. Neow is off unless
    ``neow`` is True / ``--jev-neow on`` (independent of EVENT; hang default).
    ``map_lowhp`` is the v1 uncertain shop/rest filter (hang default on).
    ``map_lowhp_hard`` is the v2 rest-then-shop override (hang default **off**).
    ``map_lowhp_soft_b`` is the elite/Boss low-HP soft bias (hang default **off**, opt-in).
    """

    phases: frozenset[str] = DEFAULT_JEV_PHASES
    event: bool = False
    neow: bool | None = None
    map_lowhp: bool = MAP_LOWHP_ON
    map_lowhp_hard: bool = MAP_LOWHP_HARD_ON
    map_lowhp_soft_b: bool = MAP_LOWHP_SOFT_B_ON

    def resolved_phases(self) -> frozenset[str]:
        phases = set(self.phases)
        if self.event:
            phases.add("event")
        return frozenset(phases)

    def allows_event(self) -> bool:
        return "event" in self.resolved_phases()

    def allows_neow(self) -> bool:
        # Hang default: Neow random unless --jev-neow on.
        if self.neow is None:
            return False
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
    map_lowhp: str | None = None,
    map_lowhp_hard: str | None = None,
    map_lowhp_soft_b: str | None = None,
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
    lowhp = MAP_LOWHP_ON if map_lowhp is None else map_lowhp == "on"
    hard = MAP_LOWHP_HARD_ON if map_lowhp_hard is None else map_lowhp_hard == "on"
    soft_b = MAP_LOWHP_SOFT_B_ON if map_lowhp_soft_b is None else map_lowhp_soft_b == "on"
    return JevPolicyFlags(
        phases=phases,
        event=event,
        neow=neow,
        map_lowhp=lowhp,
        map_lowhp_hard=hard,
        map_lowhp_soft_b=soft_b,
    )


__all__ = [
    "DEFAULT_JEV_FLAGS",
    "JevPolicyFlags",
    "parse_jev_phases",
    "resolve_jev_flags",
]
