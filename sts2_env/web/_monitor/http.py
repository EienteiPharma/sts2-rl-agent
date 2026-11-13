from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from sts2_env.web._monitor.cli import port_candidates
from sts2_env.web._monitor.clock import Clock, WallClock
from sts2_env.web._monitor.ledger import MonitorRuntime
from sts2_env.web._monitor.scan import scan_watch_root
from sts2_env.web._monitor.types import LOOPBACK, LoopbackBind, MonitorConfig, Port, RunId

_STATIC = Path(__file__).resolve().parent.parent / "static"
_RUNTIME: MonitorRuntime | None = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


def snapshot_to_json(snap) -> bytes:
    data = asdict(snap)
    return json.dumps(data, default=_json_default).encode("utf-8")


class _MonitorHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "sts2-train-monitor/1"

    def log_message(self, format: str, *args) -> None:
        return

    def _route(self) -> str:
        return urlparse(self.path).path

    def do_GET(self) -> None:
        path = self._route()
        if path in ("/", "/index.html"):
            self._serve_file(_STATIC / "train_monitor.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (_STATIC / rel).resolve()
            if not str(target).startswith(str(_STATIC.resolve())):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if target.is_file():
                ctype = "application/javascript" if target.suffix == ".js" else "text/plain"
                self._serve_file(target, ctype)
                return
        if path == "/api/snapshot":
            runtime = _RUNTIME
            if runtime is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            qs = parse_qs(urlparse(self.path).query)
            raw_id = (qs.get("run_id") or [None])[0]
            run_id = RunId(raw_id) if raw_id else None
            known = {row.run_id for row in runtime.snapshot().runs}
            if run_id is not None and run_id not in known:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = snapshot_to_json(runtime.snapshot(run_id))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/events":
            self._serve_sse()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

    def _serve_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_sse(self) -> None:
        runtime = _RUNTIME
        assert runtime is not None
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        snap = runtime.snapshot()
        payload = {"kind": "snapshot", "seq": 0, "snapshot": json.loads(snapshot_to_json(snap))}
        try:
            self.wfile.write(f"event: snapshot\ndata: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()
            for item in runtime.ledger.subscribe():
                if item.snapshot is not None:
                    payload = {
                        "kind": "snapshot",
                        "seq": int(item.seq),
                        "snapshot": json.loads(snapshot_to_json(item.snapshot)),
                    }
                    self.wfile.write(
                        f"event: snapshot\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    continue
                ev = item.event
                if ev is None:
                    continue
                payload = {"kind": "event", "seq": int(item.seq), "event": asdict(ev)}
                self.wfile.write(
                    f"event: event\ndata: {json.dumps(payload, default=_json_default)}\n\n".encode(
                        "utf-8"
                    )
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


def _listen(preferred: Port | None) -> tuple[LoopbackBind, ThreadingHTTPServer]:
    last_err: OSError | None = None
    for port in port_candidates(preferred):
        try:
            server = _MonitorHTTPServer((LOOPBACK, port), MonitorHandler)
            return LoopbackBind(host="127.0.0.1", port=Port(port)), server
        except OSError as exc:
            last_err = exc
    raise RuntimeError("no free port in 32100-32199") from last_err


def create_monitor_server(
    cfg: MonitorConfig, clock: Clock | None = None
) -> tuple[LoopbackBind, ThreadingHTTPServer, MonitorRuntime]:
    global _RUNTIME
    clk = clock or WallClock()
    bind, server = _listen(cfg.preferred_port)
    runtime = MonitorRuntime(cfg, clk)
    now = clk.now()
    index, events = scan_watch_root(cfg.watch_root, runtime.cursors, now)
    runtime.ledger.apply(events, index, cfg, now)
    runtime.start()
    _RUNTIME = runtime
    return bind, server, runtime


def serve(cfg: MonitorConfig, clock: Clock | None = None) -> LoopbackBind:
    bind, server, runtime = create_monitor_server(cfg, clock)
    try:
        server.serve_forever()
    finally:
        runtime.stop()
        server.server_close()
    return bind
