from __future__ import annotations

import os
from pathlib import Path

import pytest

from sts2_env.web._monitor.cli import bind_loopback, parse_args
from sts2_env.web._monitor.types import PORT_RANGE, Port

_FIXTURES = Path(__file__).resolve().parents[1] / "sts2_env" / "web" / "_monitor" / "fixtures"


def test_test_mode_incompatible_with_watch_root(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--test-mode", "--watch-root", str(tmp_path)])
    assert exc.value.code == 2


def test_test_mode_uses_fixtures():
    cfg = parse_args(["--test-mode"])
    assert cfg.test_mode is True
    assert cfg.watch_root == _FIXTURES


def test_refresh_fixture_mtimes_touches_directories(tmp_path: Path):
    from sts2_env.web._monitor.cli import _refresh_fixture_mtimes

    run = tmp_path / "train_waiting"
    logs = run / "tb_logs"
    logs.mkdir(parents=True)
    (logs / ".keep").write_text("", encoding="utf-8")
    old = 1_700_000_000.0
    os.utime(run, (old, old))
    os.utime(logs, (old, old))
    _refresh_fixture_mtimes(tmp_path)
    assert run.stat().st_mtime > old + 1_000
    assert logs.stat().st_mtime > old + 1_000


def test_bind_loopback_port_range():
    bind = bind_loopback(Port(32100))
    assert int(bind.port) in PORT_RANGE


def test_test_mode_snapshot_flag():
    from datetime import datetime, timezone

    from sts2_env.web._monitor.clock import FrozenClock
    from sts2_env.web.train_monitor import snapshot

    now = datetime(2026, 11, 12, 13, 0, 0, tzinfo=timezone.utc)
    cfg = parse_args(["--test-mode"])
    snap = snapshot(cfg, FrozenClock(now))
    assert snap.test_mode is True


def test_watch_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STS2_MONITOR_ROOT", str(tmp_path))
    cfg = parse_args([])
    assert cfg.watch_root == tmp_path
