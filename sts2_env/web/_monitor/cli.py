from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Sequence

from sts2_env.web._monitor.types import (
    LOOPBACK,
    PORT_RANGE,
    MonitorConfig,
    Port,
    Ttl,
    WatchRoot,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def port_candidates(preferred: Port | None) -> list[int]:
    order: list[int] = []
    if preferred is not None:
        order.append(int(preferred))
    order.extend(p for p in PORT_RANGE if p not in order)
    return order


def bind_loopback(preferred: Port | None) -> "LoopbackBind":
    """Reserve one loopback port without SO_REUSEADDR.

    The HTTP server later binds the same port. A probe that sets SO_REUSEADDR
    can report a busy Windows port as free.
    """
    from sts2_env.web._monitor.types import LoopbackBind

    last_err: OSError | None = None
    for port in port_candidates(preferred):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((LOOPBACK, port))
            sock.close()
            return LoopbackBind(host="127.0.0.1", port=Port(port))
        except OSError as exc:
            last_err = exc
            sock.close()
    raise RuntimeError("no free port in 32100-32199") from last_err


def _refresh_fixture_mtimes(root: Path) -> None:
    now = time.time()
    files: list[Path] = []
    dirs: list[Path] = [root]
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            dirs.append(path)
    for path in files:
        os.utime(path, (now, now))
    for path in dirs:
        os.utime(path, (now, now))


def parse_args(argv: Sequence[str] | None = None) -> MonitorConfig:
    parser = argparse.ArgumentParser(prog="sts2_env.web.train_monitor")
    parser.add_argument("--watch-root", type=Path, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ttl-seconds", type=float, default=120.0)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--declare-timesteps", type=int, default=None)
    ns = parser.parse_args(list(argv) if argv is not None else None)

    if ns.test_mode and ns.watch_root is not None:
        print("--test-mode is incompatible with --watch-root", file=sys.stderr)
        raise SystemExit(2)

    if ns.test_mode:
        watch = _FIXTURES
        _refresh_fixture_mtimes(watch)
    else:
        env_root = os.environ.get("STS2_MONITOR_ROOT")
        watch = ns.watch_root or (Path(env_root) if env_root else None)
        if watch is None:
            parser.error("--watch-root or STS2_MONITOR_ROOT is required")
        WatchRoot(watch)

    port = Port(ns.port) if ns.port is not None else None
    if port is not None and int(port) not in PORT_RANGE:
        print("port must be in 32100-32199", file=sys.stderr)
        raise SystemExit(2)

    return MonitorConfig(
        watch_root=watch,
        ttl=Ttl(ns.ttl_seconds),
        test_mode=bool(ns.test_mode),
        preferred_port=port,
        declared_timesteps=ns.declare_timesteps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from sts2_env.web._monitor.http import create_monitor_server

    cfg = parse_args(argv)
    bind, server, runtime = create_monitor_server(cfg, None)
    print(bind.url(), flush=True)
    try:
        server.serve_forever()
    finally:
        runtime.stop()
        server.server_close()
    return 0
