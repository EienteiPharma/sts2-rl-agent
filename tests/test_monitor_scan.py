from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sts2_env.web._monitor.scan import (
    discover,
    scan_logs,
    scan_train,
    scan_watch_root,
)
from sts2_env.web._monitor.tfevents import build_event_payload, wrap_tfrecord
from sts2_env.web._monitor.types import OverlayId, ScanCursors, WatchRoot

UTC = timezone.utc
NOW = datetime(2026, 11, 12, 13, 0, 0, tzinfo=UTC)


def _train_waiting(tmp_path: Path) -> Path:
    run = tmp_path / "run_wait"
    run.mkdir()
    (run / "tb_logs").mkdir()
    return run


def test_discover_train_dir(tmp_path: Path):
    run = _train_waiting(tmp_path)
    found = discover(tmp_path)
    assert run in found


def test_discover_collect_and_gate_dirs(tmp_path: Path):
    collect = tmp_path / "collect_running" / "w0"
    collect.mkdir(parents=True)
    np.savez(collect / "seed_0.npz", x=np.array([1]))
    gate = tmp_path / "gate_running" / "shards"
    gate.mkdir(parents=True)
    np.savez(gate / "s0.npz", x=np.array([1]))
    found = discover(tmp_path)
    assert (tmp_path / "collect_running") in found
    assert (tmp_path / "gate_running") in found


def test_foreign_monitor_csv_ignored(tmp_path: Path):
    run = tmp_path / "run_a"
    run.mkdir()
    mon = run / "monitor"
    mon.mkdir()
    foreign = mon / "train.monitor.csv"
    foreign.write_text("step,reward\n1,0\n", encoding="utf-8")
    tb = run / "tb_logs" / "MaskablePPO_0"
    tb.mkdir(parents=True)
    evf = tb / "events.out.tfevents.0"
    evf.write_bytes(wrap_tfrecord(build_event_payload(5, "rollout/ep_rew_mean", 0.1)))
    cursors = ScanCursors()
    evidence, events = scan_train(run, tmp_path, cursors, NOW)
    assert evidence.tb_max_step == 5
    assert evidence.tb_last_decode_at is not None
    assert "WORKER_FAIL" not in evidence.log_tokens
    tb_deliveries = [e for e in events if e.type == "delivery" and e.source == "tfevents"]
    assert tb_deliveries
    assert all(e.edge_from is None and e.edge_to is None for e in tb_deliveries)


def test_path_escape_rejected(tmp_path: Path):
    root = tmp_path / "watch"
    root.mkdir()
    wr = WatchRoot(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    with __import__("pytest").raises(ValueError):
        wr.rel(outside)


def test_zip_first_seen_emits_process_delivery(tmp_path: Path):
    run = tmp_path / "run_zip"
    run.mkdir()
    (run / "tb_logs").mkdir()
    (run / "final_model.zip").write_bytes(b"PK")
    cursors = ScanCursors()
    _, events = scan_train(run, tmp_path, cursors, NOW)
    dels = [e for e in events if e.type == "delivery" and e.source == "checkpoint_zip"]
    assert len(dels) == 1
    assert dels[0].edge_from == "train-go"
    assert dels[0].edge_to == "train-zips"
    _, events2 = scan_train(run, tmp_path, cursors, NOW)
    assert not any(e.type == "delivery" and e.source == "checkpoint_zip" for e in events2)


def test_eval_npz_not_delivery_when_unchanged(tmp_path: Path):
    run = tmp_path / "run_npz"
    run.mkdir()
    (run / "tb_logs").mkdir()
    el = run / "eval_logs"
    el.mkdir()
    npz = el / "evaluations.npz"
    np.savez(npz, timesteps=np.array([1, 2]), results=np.zeros(2), ep_lengths=np.zeros(2))
    cursors = ScanCursors()
    scan_train(run, tmp_path, cursors, NOW)
    _, events2 = scan_train(run, tmp_path, cursors, NOW)
    assert not any(e.type == "delivery" and e.source == "eval_npz" for e in events2)


def test_scan_watch_root_emits_time_elapsed(tmp_path: Path):
    _train_waiting(tmp_path)
    _, events = scan_watch_root(tmp_path, ScanCursors(), NOW)
    assert any(e.type == "time_elapsed" for e in events)


def test_second_scan_keeps_failure_token(tmp_path: Path):
    path = tmp_path / "train.log"
    path.write_text("WORKER_FAIL\n", encoding="utf-8")
    cur, tokens, _ = scan_logs(path, None, OverlayId("train-go"), None, NOW)
    assert "WORKER_FAIL" in tokens
    cur2, tokens2, _ = scan_logs(path, cur, OverlayId("train-go"), None, NOW)
    assert "WORKER_FAIL" in tokens2


def test_failed_log_fixture_is_failure_token():
    from datetime import datetime, timezone

    from sts2_env.web._monitor.fold import status_train
    from sts2_env.web._monitor.types import Ttl

    root = Path("sts2_env/web/_monitor/fixtures")
    run = root / "train_failed"
    if not (run / "train.log").is_file():
        return
    ev, _ = scan_train(run, root, ScanCursors(), datetime.now(timezone.utc))
    st, reason = status_train(ev, datetime.now(timezone.utc), Ttl(120.0))
    assert "WORKER_FAIL" in ev.log_tokens
    assert st == "failed"
    assert reason == "failure_token"


def test_log_shrink_rereads_from_start(tmp_path: Path):
    path = tmp_path / "train.log"
    path.write_text("prefix WORKER_FAIL suffix\n", encoding="utf-8")
    cur, tokens, _fails = scan_logs(path, None, OverlayId("train-go"), None, NOW)
    assert "WORKER_FAIL" in tokens
    path.write_text("TRAIN_OK\n", encoding="utf-8")
    assert cur.byte_offset > path.stat().st_size
    cur2, tokens2, _fails2 = scan_logs(path, cur, OverlayId("train-go"), None, NOW)
    assert "TRAIN_OK" in tokens2
    assert cur2.byte_offset == len(path.read_bytes())
