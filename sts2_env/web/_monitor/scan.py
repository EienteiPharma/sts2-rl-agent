from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sts2_env.web._monitor.tfevents import parse_tfevents
from sts2_env.web._monitor.types import (
    CollectEvidence,
    Delivery,
    DeliveryId,
    EvidenceIndex,
    FAILURE_LOG_TOKENS,
    FileCursor,
    GateCEvidence,
    LogCursor,
    LogToken,
    MonitorEvent,
    ObservedFact,
    OverlayId,
    Pulse,
    PulseSource,
    RunId,
    ScanCursors,
    TimeElapsed,
    TrainEvidence,
    WatchRoot,
    ZipPresence,
)

_SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__"}
_MAX_DEPTH = 8
_LOG_RE = re.compile(r"\b(" + "|".join(t.value for t in LogToken) + r")\b")


def discover(watch_root: Path) -> tuple[Path, ...]:
    root = WatchRoot(watch_root).path
    found: list[Path] = []

    def walk(dir_path: Path, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        try:
            entries = list(dir_path.iterdir())
        except OSError:
            return
        if _is_train_dir(dir_path):
            found.append(dir_path)
            return
        if _is_collect_dir(dir_path):
            found.append(dir_path)
            return
        if _is_gate_dir(dir_path):
            found.append(dir_path)
            return
        for child in entries:
            if child.is_dir() and child.name not in _SKIP_DIRS:
                walk(child, depth + 1)

    walk(root, 0)
    return tuple(dict.fromkeys(found))


def _is_train_dir(p: Path) -> bool:
    if (p / "tb_logs").is_dir():
        return True
    if (p / "best_model" / "best_model.zip").is_file():
        return True
    if (p / "final_model.zip").is_file():
        return True
    return False


def _is_collect_dir(p: Path) -> bool:
    if (p / "todo").is_dir():
        return True
    return any(p.glob("w*/seed_*.npz"))


def _is_gate_dir(p: Path) -> bool:
    if (p / "shards").is_dir():
        return True
    if (p / "dataset.npz").is_file():
        return True
    if list(p.glob("eval/w*.log")):
        return True
    return False


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _rel(watch_root: Path, path: Path) -> str:
    return WatchRoot(watch_root).rel(path)


def read_eval_npz(path: Path) -> tuple[int, datetime] | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "timesteps" not in z.files:
                return None
            rows = int(len(z["timesteps"]))
        return rows, _mtime(path)
    except (OSError, ValueError, zipfile.BadZipFile):
        return None


def scan_logs(
    path: Path, cursor: LogCursor | None, overlay_id: OverlayId, run_id: RunId | None, now: datetime
) -> tuple[LogCursor, tuple[str, ...], tuple]:
    from sts2_env.web._monitor.types import FailureToken

    try:
        data = path.read_bytes()
    except OSError:
        kept = () if cursor is None else cursor.tokens
        cur = LogCursor(str(path), 0 if cursor is None else cursor.byte_offset, kept)
        return cur, kept, ()
    start = 0 if cursor is None else cursor.byte_offset
    # A rotated or truncated log is shorter than the old offset. Clamping to
    # EOF would skip every token in the replacement file.
    if start > len(data):
        start = 0
    prev = () if cursor is None or start == 0 else cursor.tokens
    chunk = data[start:]
    tokens: list[str] = []
    failures = []
    for match in _LOG_RE.finditer(chunk.decode("utf-8", "replace")):
        tok = match.group(1)
        if tok not in tokens:
            tokens.append(tok)
        if tok in FAILURE_LOG_TOKENS:
            failures.append(
                FailureToken(
                    type="failure_token",
                    overlay_id=overlay_id,
                    token=tok,
                    path=_safe_rel_path(path),
                    observed_at=now,
                    run_id=run_id,
                )
            )
    merged = tuple(dict.fromkeys(prev + tuple(tokens)))
    new_cur = LogCursor(str(path), len(data), merged)
    return new_cur, merged, tuple(failures)


def _safe_rel_path(path: Path) -> str:
    return path.as_posix()


def _tfevent_paths(run_dir: Path) -> list[Path]:
    tb = run_dir / "tb_logs"
    if not tb.is_dir():
        return []
    return sorted(tb.rglob("events.out.tfevents.*"))


def scan_train(
    run_dir: Path, watch_root: Path, cursors: ScanCursors, now: datetime
) -> tuple[TrainEvidence, tuple[MonitorEvent, ...]]:
    relpath = _rel(watch_root, run_dir)
    run_id = RunId(relpath.replace("/", "_") or "train")
    events: list[MonitorEvent] = []
    tb_max_step: int | None = None
    tb_last: datetime | None = None
    tb_scalars: list[ObservedFact] = []
    key_base = str(run_dir)

    for tf_path in _tfevent_paths(run_dir):
        cur = cursors.tfevents.get(str(tf_path))
        new_cur, facts, delivered = parse_tfevents(tf_path, cur)
        cursors.tfevents[str(tf_path)] = new_cur
        if new_cur.max_step:
            tb_max_step = max(tb_max_step or 0, new_cur.max_step)
            file_at = _mtime(tf_path)
            tb_last = file_at if tb_last is None else max(tb_last, file_at)
        if facts:
            dated = [f.at for f in facts if f.at is not None]
            if dated:
                tb_last = max(dated) if tb_last is None else max([tb_last, *dated])
        tb_scalars.extend(f for f in facts if f.kind in ("tb_scalar", "tb_fps"))
        if delivered:
            events.append(
                Delivery(
                    type="delivery",
                    delivery_id=DeliveryId(f"{run_id}:tb:{new_cur.byte_offset}"),
                    overlay_id=OverlayId("train-go"),
                    edge_from=None,
                    edge_to=None,
                    source="tfevents",
                    observed_at=now,
                    run_id=run_id,
                    facts=tuple(facts),
                )
            )

    eval_path = run_dir / "eval_logs" / "evaluations.npz"
    eval_rows = None
    eval_mtime = None
    if eval_path.is_file():
        got = read_eval_npz(eval_path)
        if got:
            eval_rows, eval_mtime = got
            fc = cursors.files.get(str(eval_path))
            st = eval_path.stat()
            if fc is None or fc.n_rows != eval_rows:
                cursors.files[str(eval_path)] = FileCursor(
                    str(eval_path), st.st_mtime_ns, st.st_size, eval_rows
                )
                events.append(
                    Delivery(
                        type="delivery",
                        delivery_id=DeliveryId(f"{run_id}:eval:{eval_rows}"),
                        overlay_id=OverlayId("train-go"),
                        edge_from=None,
                        edge_to=None,
                        source="eval_npz",
                        observed_at=now,
                        run_id=run_id,
                        facts=(ObservedFact(kind="npz_rows", n_rows=eval_rows, at=eval_mtime),),
                    )
                )

    def zip_presence(rel: str) -> ZipPresence | None:
        p = run_dir / rel
        if not p.is_file():
            return None
        return ZipPresence(rel, _mtime(p), p.stat().st_size)

    best = zip_presence("best_model/best_model.zip")
    final = zip_presence("final_model.zip")
    for label, z, source in (
        ("best", best, "best_zip"),
        ("final", final, "final_zip"),
    ):
        if z is None:
            continue
        fc = cursors.files.get(str(run_dir / z.relpath))
        st = (run_dir / z.relpath).stat()
        if fc is None or fc.size != st.st_size or fc.mtime_ns != st.st_mtime_ns:
            cursors.files[str(run_dir / z.relpath)] = FileCursor(
                str(run_dir / z.relpath), st.st_mtime_ns, st.st_size
            )
            events.append(
                Pulse(
                    type="pulse",
                    overlay_id=OverlayId("train-zips"),
                    source=source,  # type: ignore[arg-type]
                    observed_at=now,
                    run_id=run_id,
                )
            )
            events.append(
                Delivery(
                    type="delivery",
                    delivery_id=DeliveryId(f"{run_id}:zip:{z.relpath}:{st.st_mtime_ns}"),
                    overlay_id=OverlayId("train-zips"),
                    edge_from=OverlayId("train-go"),
                    edge_to=OverlayId("train-zips"),
                    source="checkpoint_zip",
                    observed_at=now,
                    run_id=run_id,
                    facts=(
                        ObservedFact(
                            kind="zip_presence",
                            path=z.relpath,
                            at=z.mtime,
                        ),
                    ),
                )
            )

    log_tokens: list[str] = []
    for log_path in sorted(run_dir.glob("**/*")):
        if not log_path.is_file():
            continue
        if log_path.suffix not in (".log", ".txt"):
            continue
        parts = log_path.relative_to(run_dir).as_posix()
        if parts == "monitor/train.monitor.csv" or log_path.name == "train.monitor.csv":
            continue
        cur = cursors.logs.get(str(log_path))
        new_cur, toks, fails = scan_logs(log_path, cur, OverlayId("train-go"), run_id, now)
        cursors.logs[str(log_path)] = new_cur
        log_tokens.extend(t for t in toks if t not in log_tokens)
        events.extend(fails)

    ev = TrainEvidence(
        run_id=run_id,
        relpath=relpath,
        dir_mtime=_mtime(run_dir),
        tb_max_step=tb_max_step,
        tb_last_decode_at=tb_last,
        tb_scalars=tuple(tb_scalars),
        eval_npz_rows=eval_rows,
        eval_npz_mtime=eval_mtime,
        best_zip=best,
        final_zip=final,
        log_tokens=tuple(log_tokens),
    )
    return ev, tuple(events)


def scan_collect(
    out_dir: Path, watch_root: Path, cursors: ScanCursors, now: datetime
) -> tuple[CollectEvidence, tuple[MonitorEvent, ...]]:
    relpath = _rel(watch_root, out_dir)
    run_id = RunId(f"collect_{relpath}")
    events: list[MonitorEvent] = []
    todo = len(list((out_dir / "todo").glob("*"))) if (out_dir / "todo").is_dir() else 0
    claimed = sum(1 for _ in out_dir.glob("w*/claimed/*")) if out_dir.glob("w*") else 0
    seeds = list(out_dir.glob("w*/seed_*.npz"))
    complete = len(seeds)
    partial = len(list(out_dir.glob("w*/seed_*.npz.tmp")))
    known_key = str(out_dir)
    known = cursors.known_seeds.get(known_key, frozenset())
    new_names = {s.name for s in seeds}
    for s in seeds:
        if s.name not in known:
            events.append(
                Delivery(
                    type="delivery",
                    delivery_id=DeliveryId(f"{run_id}:seed:{s.name}"),
                    overlay_id=OverlayId("fg4"),
                    edge_from=None,
                    edge_to=None,
                    source="seed_npz",
                    observed_at=now,
                    run_id=run_id,
                    facts=(ObservedFact(kind="npz_rows", path=s.name, n_rows=1),),
                )
            )
    cursors.known_seeds[known_key] = frozenset(new_names)
    last_seed = max((_mtime(s) for s in seeds), default=None)
    log_tokens: list[str] = []
    for log_path in sorted(out_dir.glob("**/*.log")):
        cur = cursors.logs.get(str(log_path))
        new_cur, toks, fails = scan_logs(log_path, cur, OverlayId("fg4"), run_id, now)
        cursors.logs[str(log_path)] = new_cur
        log_tokens.extend(t for t in toks if t not in log_tokens)
        events.extend(fails)

    ev = CollectEvidence(
        run_id=run_id,
        relpath=relpath,
        todo_count=todo,
        claimed_count=claimed,
        complete_npz=complete,
        partial_npz=partial,
        last_seed_at=last_seed,
        log_tokens=tuple(log_tokens),
    )
    return ev, tuple(events)


def scan_gate_c(
    out_dir: Path, watch_root: Path, cursors: ScanCursors, now: datetime
) -> tuple[GateCEvidence, tuple[MonitorEvent, ...]]:
    relpath = _rel(watch_root, out_dir)
    run_id = RunId(f"gate_{relpath}")
    events: list[MonitorEvent] = []
    shards = list((out_dir / "shards").glob("*.npz")) if (out_dir / "shards").is_dir() else []
    shard_key = str(out_dir / "shards")
    known = cursors.known_seeds.get(shard_key, frozenset())
    for sh in shards:
        if sh.name not in known:
            events.append(
                Delivery(
                    type="delivery",
                    delivery_id=DeliveryId(f"{run_id}:shard:{sh.name}"),
                    overlay_id=OverlayId("fg7"),
                    edge_from=None,
                    edge_to=None,
                    source="gate_shard",
                    observed_at=now,
                    run_id=run_id,
                    facts=(),
                )
            )
    cursors.known_seeds[shard_key] = frozenset(s.name for s in shards)
    last_shard = max((_mtime(s) for s in shards), default=None)
    dataset = out_dir / "dataset.npz"
    has_dataset = dataset.is_file()
    dataset_mtime = _mtime(dataset) if has_dataset else None
    if has_dataset:
        fc = cursors.files.get(str(dataset))
        st = dataset.stat()
        if fc is None or fc.mtime_ns != st.st_mtime_ns:
            cursors.files[str(dataset)] = FileCursor(str(dataset), st.st_mtime_ns, st.st_size)
            events.append(
                Delivery(
                    type="delivery",
                    delivery_id=DeliveryId(f"{run_id}:dataset"),
                    overlay_id=OverlayId("fg7"),
                    edge_from=None,
                    edge_to=None,
                    source="dataset_npz",
                    observed_at=now,
                    run_id=run_id,
                    facts=(),
                )
            )
    ce_path = out_dir / "ce_model.zip"
    ce_zip = None
    if ce_path.is_file():
        ce_zip = ZipPresence("ce_model.zip", _mtime(ce_path), ce_path.stat().st_size)
        fc = cursors.files.get(str(ce_path))
        st = ce_path.stat()
        if fc is None or fc.size != st.st_size:
            cursors.files[str(ce_path)] = FileCursor(str(ce_path), st.st_mtime_ns, st.st_size)
            events.append(
                Pulse(
                    type="pulse",
                    overlay_id=OverlayId("fg7"),
                    source="ce_zip",
                    observed_at=now,
                    run_id=run_id,
                )
            )
    log_tokens: list[str] = []
    for log_path in sorted(out_dir.glob("eval/w*.log")) + sorted(out_dir.glob("*.log")):
        cur = cursors.logs.get(str(log_path))
        new_cur, toks, fails = scan_logs(log_path, cur, OverlayId("fg7"), run_id, now)
        cursors.logs[str(log_path)] = new_cur
        log_tokens.extend(t for t in toks if t not in log_tokens)
        events.extend(fails)

    ev = GateCEvidence(
        run_id=run_id,
        relpath=relpath,
        shard_count=len(shards),
        last_shard_at=last_shard,
        has_dataset=has_dataset,
        dataset_mtime=dataset_mtime,
        ce_zip=ce_zip,
        log_tokens=tuple(log_tokens),
    )
    return ev, tuple(events)


def scan_watch_root(
    watch_root: Path, cursors: ScanCursors, now: datetime
) -> tuple[EvidenceIndex, tuple[MonitorEvent, ...]]:
    wr = WatchRoot(watch_root)
    paths = discover(wr.path)
    trains: list[TrainEvidence] = []
    collects: list[CollectEvidence] = []
    gates: list[GateCEvidence] = []
    events: list[MonitorEvent] = []
    for p in paths:
        if _is_train_dir(p):
            ev, evs = scan_train(p, wr.path, cursors, now)
            trains.append(ev)
            events.extend(evs)
        elif _is_gate_dir(p):
            ev, evs = scan_gate_c(p, wr.path, cursors, now)
            gates.append(ev)
            events.extend(evs)
        elif _is_collect_dir(p):
            ev, evs = scan_collect(p, wr.path, cursors, now)
            collects.append(ev)
            events.extend(evs)
    events.append(TimeElapsed(type="time_elapsed", observed_at=now))
    return EvidenceIndex(tuple(trains), tuple(collects), tuple(gates)), tuple(events)
