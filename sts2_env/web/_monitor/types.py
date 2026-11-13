from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Literal, NewType, Protocol, Sequence

Status = Literal["running", "waiting", "complete", "failed", "stale", "unknown"]
DomKind = Literal["node", "step"]
Lane = Literal["train", "collect", "gate_c", "legend"]
MetricKind = Literal["tb_step", "eval_episodes", "seed_count", "shard_count", "zip_bytes"]
FactKind = Literal[
    "tb_step",
    "tb_fps",
    "tb_scalar",
    "npz_rows",
    "zip_presence",
    "last_advance_at",
    "log_token",
    "unobservable",
    "operator_declared",
]
DeliverySource = Literal[
    "tfevents",
    "eval_npz",
    "seed_npz",
    "gate_shard",
    "dataset_npz",
    "checkpoint_zip",
]
PulseSource = Literal["best_zip", "final_zip", "ce_zip"]
StatusReason = Literal[
    "stream_fresh",
    "parent_waiting",
    "terminal_artifact",
    "failure_token",
    "ttl_expired",
    "no_evidence",
    "unobservable",
    "static_legend",
]

OverlayId = NewType("OverlayId", str)
RunId = NewType("RunId", str)
EventSeq = NewType("EventSeq", int)
DeliveryId = NewType("DeliveryId", str)

Port = NewType("Port", int)
PORT_RANGE = range(32100, 32200)
LOOPBACK = "127.0.0.1"

Instant = datetime


class LogToken(StrEnum):
    WORKER_FAIL = "WORKER_FAIL"
    B4_REFUSE = "B4_REFUSE"
    SHARD_OK = "SHARD_OK"
    COLLECT_OK = "COLLECT_OK"
    TRAIN_OK = "TRAIN_OK"
    EVAL_OK = "EVAL_OK"
    PAIRED = "PAIRED"
    VERDICT = "VERDICT"
    B4_PENDING = "B4_PENDING"
    B4_SHARD = "B4_SHARD"


FAILURE_LOG_TOKENS = frozenset({LogToken.WORKER_FAIL, LogToken.B4_REFUSE})


class Clock(Protocol):
    def now(self) -> Instant:
        ...


@dataclass(frozen=True)
class Ttl:
    seconds: float

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("ttl must be > 0")

    def window(self) -> timedelta:
        return timedelta(seconds=self.seconds)


@dataclass(frozen=True)
class WatchRoot:
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _resolve_watch_dir(self.path))

    def join(self, *parts: str) -> Path:
        candidate = self.path.joinpath(*parts)
        _reject_escape(self.path, candidate)
        return candidate

    def rel(self, child: Path) -> str:
        resolved = child.resolve()
        _reject_escape(self.path, resolved)
        return resolved.relative_to(self.path).as_posix()


def _resolve_watch_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"watch root is not a directory: {resolved}")
    return resolved


def _reject_escape(root: Path, candidate: Path) -> None:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes watch root: {candidate}") from exc
    if candidate.drive and root.drive and candidate.drive.lower() != root.drive.lower():
        raise ValueError(f"path escapes watch root via drive: {candidate}")


@dataclass(frozen=True)
class LoopbackBind:
    host: Literal["127.0.0.1"]
    port: Port

    def __post_init__(self) -> None:
        if self.host != LOOPBACK:
            raise ValueError("host must be 127.0.0.1")
        if int(self.port) not in PORT_RANGE:
            raise ValueError(f"port must be in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")

    def url(self) -> str:
        return f"http://{self.host}:{int(self.port)}/"


@dataclass(frozen=True)
class MonitorConfig:
    watch_root: Path
    ttl: Ttl
    test_mode: bool
    preferred_port: Port | None
    declared_timesteps: int | None

    def __post_init__(self) -> None:
        if not self.test_mode:
            _resolve_watch_dir(self.watch_root)
        if self.preferred_port is not None and int(self.preferred_port) not in PORT_RANGE:
            raise ValueError(f"port must be in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")

    def cell_ttl_cutoff(self, now: Instant) -> Instant:
        return now - self.ttl.window()


@dataclass(frozen=True)
class ObservedFact:
    kind: FactKind
    tag: str | None = None
    step: int | None = None
    value: float | None = None
    at: Instant | None = None
    detail: str | None = None
    n_rows: int | None = None
    path: str | None = None
    token: str | None = None


@dataclass(frozen=True)
class OverlayCell:
    overlay_id: OverlayId
    dom_kind: DomKind
    status: Status
    reason: StatusReason
    pulse: bool
    facts: tuple[ObservedFact, ...]


@dataclass(frozen=True)
class LabeledMetric:
    kind: MetricKind
    label: str
    value: int
    unit: str
    total: int | None = None


@dataclass(frozen=True)
class Occupant:
    run_id: RunId
    lane: Lane
    overlay_id: OverlayId
    substep_id: str
    substep_label: str
    status: Status
    reason: StatusReason
    metrics: tuple[LabeledMetric, ...]
    last_advance_at: Instant | None


@dataclass(frozen=True)
class RunRosterRow:
    run_id: RunId
    lane: Lane
    relpath: str
    status: Status
    tb_max_step: int | None
    has_final_zip: bool
    last_advance_at: Instant | None


@dataclass(frozen=True)
class LegendItem:
    id: str
    text: str
    not_a_verdict: Literal[True] = True


@dataclass(frozen=True)
class StaticLegend:
    kind: Literal["static_constants"]
    items: tuple[LegendItem, ...]


@dataclass(frozen=True)
class ActivityEdge:
    edge_from: OverlayId
    edge_to: OverlayId
    active: bool
    missing_signal: str | None


@dataclass(frozen=True)
class RecentDelivery:
    delivery_id: DeliveryId
    overlay_id: OverlayId
    edge_from: OverlayId | None
    edge_to: OverlayId | None
    source: DeliverySource
    observed_at: Instant


@dataclass(frozen=True)
class Snapshot:
    schema: Literal[1]
    test_mode: bool
    watch_root: str
    generated_at: Instant
    ttl_seconds: float
    active_run_id: RunId | None
    cells: tuple[OverlayCell, ...]
    runs: tuple[RunRosterRow, ...]
    occupants: tuple[Occupant, ...]
    activity_edges: tuple[ActivityEdge, ...]
    legend: StaticLegend
    recent_deliveries: tuple[RecentDelivery, ...]
    declared_timesteps: int | None

    def cell(self, overlay_id: str) -> OverlayCell:
        for c in self.cells:
            if c.overlay_id == overlay_id:
                return c
        raise KeyError(overlay_id)


@dataclass(frozen=True)
class Delivery:
    type: Literal["delivery"]
    delivery_id: DeliveryId
    overlay_id: OverlayId
    edge_from: OverlayId | None
    edge_to: OverlayId | None
    source: DeliverySource
    observed_at: Instant
    run_id: RunId
    facts: tuple[ObservedFact, ...]


@dataclass(frozen=True)
class Pulse:
    type: Literal["pulse"]
    overlay_id: OverlayId
    source: PulseSource
    observed_at: Instant
    run_id: RunId


@dataclass(frozen=True)
class StatusChanged:
    type: Literal["status_changed"]
    overlay_id: OverlayId
    status: Status
    reason: StatusReason
    observed_at: Instant


@dataclass(frozen=True)
class FailureToken:
    type: Literal["failure_token"]
    overlay_id: OverlayId
    token: str
    path: str
    observed_at: Instant
    run_id: RunId | None


@dataclass(frozen=True)
class TimeElapsed:
    type: Literal["time_elapsed"]
    observed_at: Instant


MonitorEvent = Delivery | Pulse | StatusChanged | FailureToken | TimeElapsed


@dataclass(frozen=True)
class LiveItem:
    seq: EventSeq
    snapshot: Snapshot | None
    event: MonitorEvent | None

    def __post_init__(self) -> None:
        if (self.snapshot is None) == (self.event is None):
            raise ValueError("LiveItem is snapshot XOR event")


@dataclass(frozen=True)
class ZipPresence:
    relpath: str
    mtime: Instant
    size: int


@dataclass(frozen=True)
class TrainEvidence:
    run_id: RunId
    relpath: str
    dir_mtime: Instant
    tb_max_step: int | None
    tb_last_decode_at: Instant | None
    tb_scalars: tuple[ObservedFact, ...]
    eval_npz_rows: int | None
    eval_npz_mtime: Instant | None
    best_zip: ZipPresence | None
    final_zip: ZipPresence | None
    log_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CollectEvidence:
    run_id: RunId
    relpath: str
    todo_count: int
    claimed_count: int
    complete_npz: int
    partial_npz: int
    last_seed_at: Instant | None
    log_tokens: tuple[str, ...]


@dataclass(frozen=True)
class GateCEvidence:
    run_id: RunId
    relpath: str
    shard_count: int
    last_shard_at: Instant | None
    has_dataset: bool
    dataset_mtime: Instant | None
    ce_zip: ZipPresence | None
    log_tokens: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceIndex:
    train: tuple[TrainEvidence, ...]
    collect: tuple[CollectEvidence, ...]
    gate_c: tuple[GateCEvidence, ...]


@dataclass(frozen=True)
class TfEventsCursor:
    path: str
    size: int
    byte_offset: int
    max_step: int


@dataclass(frozen=True)
class FileCursor:
    path: str
    mtime_ns: int
    size: int
    n_rows: int | None = None


@dataclass(frozen=True)
class LogCursor:
    path: str
    byte_offset: int
    tokens: tuple[str, ...] = ()


@dataclass
class ScanCursors:
    tfevents: dict[str, TfEventsCursor] = field(default_factory=dict)
    files: dict[str, FileCursor] = field(default_factory=dict)
    logs: dict[str, LogCursor] = field(default_factory=dict)
    known_seeds: dict[str, frozenset[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class OverlayBinding:
    overlay_id: OverlayId
    dom_kind: DomKind
    lane: Lane
    observable: bool
    edge_from: OverlayId | None
    edge_to: OverlayId | None
    note: str
