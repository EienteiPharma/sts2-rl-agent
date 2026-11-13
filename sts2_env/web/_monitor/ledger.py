from __future__ import annotations

import threading
import time
from collections import deque
from datetime import timedelta
from typing import Iterator, Sequence

from sts2_env.web._monitor.clock import Clock, WallClock
from sts2_env.web._monitor.fold import diff_status, fold
from sts2_env.web._monitor.scan import scan_watch_root
from dataclasses import replace

from sts2_env.web._monitor.types import (
    EventSeq,
    EvidenceIndex,
    LiveItem,
    MonitorConfig,
    MonitorEvent,
    RecentDelivery,
    RunId,
    ScanCursors,
    Snapshot,
)


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = Snapshot(
            schema=1,
            test_mode=False,
            watch_root="",
            generated_at=WallClock().now(),
            ttl_seconds=120.0,
            active_run_id=None,
            cells=(),
            runs=(),
            occupants=(),
            activity_edges=(),
            legend=__import__(
                "sts2_env.web._monitor.legend", fromlist=["static_legend"]
            ).static_legend(),
            recent_deliveries=(),
            declared_timesteps=None,
        )
        self._seq = EventSeq(0)
        self._events: deque[MonitorEvent] = deque()
        self._deliveries: deque[RecentDelivery] = deque()
        self._subscribers: list[deque[LiveItem]] = []
        self._index: EvidenceIndex | None = None
        self._cfg: MonitorConfig | None = None

    def apply(
        self,
        events: Sequence[MonitorEvent],
        index: EvidenceIndex,
        cfg: MonitorConfig,
        now,
    ) -> tuple[MonitorEvent, ...]:
        from sts2_env.web._monitor.fold import fold as fold_fn

        prev = self._snapshot
        snap = fold_fn(index, cfg, now)
        for d in events:
            if d.type == "delivery":
                self._deliveries.append(
                    RecentDelivery(
                        d.delivery_id,
                        d.overlay_id,
                        d.edge_from,
                        d.edge_to,
                        d.source,
                        d.observed_at,
                    )
                )
        cutoff = now - timedelta(seconds=8)
        while self._deliveries and self._deliveries[0].observed_at < cutoff:
            self._deliveries.popleft()
        snap = replace(snap, recent_deliveries=tuple(self._deliveries))
        status_events = diff_status(prev, snap, now)
        out = tuple(events) + status_events
        with self._lock:
            self._snapshot = snap
            self._index = index
            self._cfg = cfg
            for ev in out:
                self._seq = EventSeq(int(self._seq) + 1)
                item = LiveItem(seq=self._seq, snapshot=None, event=ev)
                self._events.append(ev)
                for sub in self._subscribers:
                    sub.append(item)
            self._seq = EventSeq(int(self._seq) + 1)
            snap_item = LiveItem(seq=self._seq, snapshot=snap, event=None)
            for sub in self._subscribers:
                sub.append(snap_item)
        return out

    def published(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def snapshot_for(self, now, run_id: RunId | None) -> Snapshot:
        with self._lock:
            index = self._index
            cfg = self._cfg
            deliveries = tuple(self._deliveries)
        if index is None or cfg is None:
            return self.published()
        snap = fold(index, cfg, now, run_id)
        return replace(snap, recent_deliveries=deliveries)

    def subscribe(self) -> Iterator[LiveItem]:
        q: deque[LiveItem] = deque()
        with self._lock:
            self._subscribers.append(q)
            self._seq = EventSeq(int(self._seq) + 1)
            q.append(LiveItem(seq=EventSeq(0), snapshot=self._snapshot, event=None))
        while True:
            while q:
                yield q.popleft()
            time.sleep(0.05)


class MonitorRuntime:
    POLL_INTERVAL = 2.0

    def __init__(self, cfg: MonitorConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self.ledger = Ledger()
        self.cursors = ScanCursors()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                now = self.clock.now()
                index, events = scan_watch_root(self.cfg.watch_root, self.cursors, now)
                self.ledger.apply(events, index, self.cfg, now)
                self._stop.wait(self.POLL_INTERVAL)

        self._thread = threading.Thread(target=loop, name="sts2-monitor-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def snapshot(self, run_id: RunId | None = None) -> Snapshot:
        if run_id is None:
            return self.ledger.published()
        return self.ledger.snapshot_for(self.clock.now(), run_id)

    def events(self) -> Iterator[LiveItem]:
        yield from self.ledger.subscribe()
