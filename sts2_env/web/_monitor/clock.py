from __future__ import annotations

from datetime import datetime, timezone

from sts2_env.web._monitor.types import Clock, Instant

__all__ = ["Clock", "FrozenClock", "Instant", "WallClock"]


def _require_aware(dt: Instant) -> Instant:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt


class FrozenClock:
    def __init__(self, at: Instant) -> None:
        self._at = _require_aware(at)

    @staticmethod
    def parse(iso: str) -> FrozenClock:
        text = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return FrozenClock(_require_aware(dt))

    def now(self) -> Instant:
        return self._at


class WallClock:
    def now(self) -> Instant:
        return datetime.now(timezone.utc)
