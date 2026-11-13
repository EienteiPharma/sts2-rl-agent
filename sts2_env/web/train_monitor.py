"""Localhost-only read-only training artifact monitor."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sts2_env.web._monitor.clock import Clock, WallClock
from sts2_env.web._monitor.types import (
    LoopbackBind,
    MonitorConfig,
    OverlayCell,
    Snapshot,
    Status,
    Ttl,
)

__all__ = [
    "MonitorConfig",
    "Snapshot",
    "OverlayCell",
    "Status",
    "snapshot",
    "serve",
    "main",
]


def snapshot(cfg: MonitorConfig, clock: Clock | None = None) -> Snapshot:
    from sts2_env.web._monitor.fold import fold
    from sts2_env.web._monitor.scan import scan_watch_root
    from sts2_env.web._monitor.types import EvidenceIndex, ScanCursors

    clk = clock or WallClock()
    now = clk.now()
    cursors = ScanCursors()
    index, _ = scan_watch_root(cfg.watch_root, cursors, now)
    return fold(index, cfg, now)


def serve(cfg: MonitorConfig, clock: Clock | None = None) -> LoopbackBind:
    from sts2_env.web._monitor.http import serve as _serve

    return _serve(cfg, clock)


def main(argv: Sequence[str] | None = None) -> int:
    from sts2_env.web._monitor.cli import main as _main

    return _main(argv)


def parse_args(argv: Sequence[str] | None = None) -> MonitorConfig:
    from sts2_env.web._monitor.cli import parse_args as _parse_args

    return _parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
