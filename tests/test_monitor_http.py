from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sts2_env.web._monitor.clock import FrozenClock
from sts2_env.web._monitor.http import MonitorHandler, create_monitor_server, snapshot_to_json
from sts2_env.web._monitor.types import MonitorConfig, Ttl
from sts2_env.web._monitor.tfevents import build_event_payload, wrap_tfrecord

NOW = datetime(2026, 11, 12, 13, 0, 0, tzinfo=timezone.utc)


def _fixture_root(tmp_path: Path) -> Path:
    run = tmp_path / "train_running"
    run.mkdir()
    tb = run / "tb_logs" / "MaskablePPO_0"
    tb.mkdir(parents=True)
    (tb / "events.out.tfevents.0").write_bytes(
        wrap_tfrecord(build_event_payload(99, "rollout/ep_rew_mean", 0.2))
    )
    return tmp_path


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(opener: urllib.request.OpenerDirector, url: str):
    last = None
    for _ in range(20):
        try:
            return opener.open(url, timeout=1)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
            threading.Event().wait(0.05)
    raise last


def test_get_snapshot_and_post_405(tmp_path: Path):
    root = _fixture_root(tmp_path)
    cfg = MonitorConfig(root, Ttl(120.0), False, None, None)
    bind, server, runtime = create_monitor_server(cfg, FrozenClock(NOW))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = _opener()
    try:
        base = bind.url().rstrip("/")
        with _get(opener, base + "/api/snapshot") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body["schema"] == 1
        req = urllib.request.Request(base + "/api/snapshot", method="POST")
        try:
            opener.open(req, timeout=1)
            assert False, "expected 405"
        except urllib.error.HTTPError as exc:
            assert exc.code == 405
        assert bind.host == "127.0.0.1"
        assert 32100 <= int(bind.port) < 32200
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()


def test_handler_has_no_do_post():
    assert not hasattr(MonitorHandler, "do_PUT")


def test_empty_watch_root_unknown(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = MonitorConfig(empty, Ttl(120.0), False, None, None)
    bind, server, runtime = create_monitor_server(cfg, FrozenClock(NOW))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = _opener()
    try:
        with _get(opener, bind.url().rstrip("/") + "/api/snapshot") as resp:
            body = json.loads(resp.read())
        assert body["runs"] == []
        statuses = {c["overlay_id"]: c["status"] for c in body["cells"]}
        assert statuses["train-go"] == "unknown"
        assert statuses["te-run"] == "unknown"
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()


def test_two_runs_and_unknown_run_id(tmp_path: Path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    for run, step in ((a, 3), (b, 9)):
        tb = run / "tb_logs" / "MaskablePPO_0"
        tb.mkdir(parents=True)
        (tb / "events.out.tfevents.0").write_bytes(
            wrap_tfrecord(build_event_payload(step, "rollout/ep_rew_mean", 0.1))
        )
    cfg = MonitorConfig(tmp_path, Ttl(120.0), False, None, None)
    bind, server, runtime = create_monitor_server(cfg, FrozenClock(NOW))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = _opener()
    try:
        base = bind.url().rstrip("/")
        with _get(opener, base + "/api/snapshot") as resp:
            body = json.loads(resp.read())
        ids = {row["run_id"] for row in body["runs"]}
        assert ids == {"run_a", "run_b"}
        req = urllib.request.Request(base + "/api/snapshot?run_id=nope")
        try:
            opener.open(req, timeout=1)
            assert False, "expected 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()


def test_sse_snapshot_on_each_connect(tmp_path: Path):
    import http.client

    root = _fixture_root(tmp_path)
    cfg = MonitorConfig(root, Ttl(120.0), False, None, None)
    bind, server, runtime = create_monitor_server(cfg, FrozenClock(NOW))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for _ in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", int(bind.port), timeout=3)
            conn.request("GET", "/api/events", headers={"Accept": "text/event-stream"})
            resp = conn.getresponse()
            buf = b""
            while b"\n\n" not in buf:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk
            conn.close()
            assert buf.startswith(b"event: snapshot")
            assert b'"schema": 1' in buf
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()


def test_sse_forwards_snapshot_after_ledger_apply(tmp_path: Path):
    import http.client

    from sts2_env.web._monitor.scan import scan_watch_root

    root = _fixture_root(tmp_path)
    cfg = MonitorConfig(root, Ttl(120.0), False, None, None)
    bind, server, runtime = create_monitor_server(cfg, FrozenClock(NOW))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", int(bind.port), timeout=5)
        conn.request("GET", "/api/events", headers={"Accept": "text/event-stream"})
        resp = conn.getresponse()
        buf = b""
        while buf.count(b"event: snapshot") < 1:
            chunk = resp.fp.read1(4096) if hasattr(resp.fp, "read1") else resp.read(4096)
            if not chunk:
                break
            buf += chunk
        before = buf.count(b"event: snapshot")
        index, events = scan_watch_root(cfg.watch_root, runtime.cursors, NOW)
        runtime.ledger.apply(events, index, cfg, NOW)
        while buf.count(b"event: snapshot") < before + 1:
            chunk = resp.fp.read1(4096) if hasattr(resp.fp, "read1") else resp.read(4096)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 2_000_000:
                break
        conn.close()
        assert buf.count(b"event: snapshot") >= before + 1, buf[:500]
        assert b'"schema": 1' in buf
    finally:
        runtime.stop()
        server.shutdown()
        server.server_close()
