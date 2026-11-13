from __future__ import annotations

from datetime import datetime, timezone

from sts2_env.web._monitor.legend import overlay_bindings, static_legend
from sts2_env.web._monitor.types import (
    CollectEvidence,
    EvidenceIndex,
    FAILURE_LOG_TOKENS,
    GateCEvidence,
    LabeledMetric,
    MonitorConfig,
    ActivityEdge,
    Occupant,
    ObservedFact,
    OverlayCell,
    OverlayId,
    RunId,
    RunRosterRow,
    Snapshot,
    Status,
    StatusChanged,
    StatusReason,
    TrainEvidence,
    Ttl,
)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _has_failure(tokens: tuple[str, ...]) -> bool:
    return any(t in FAILURE_LOG_TOKENS for t in tokens)


def _within_ttl(at, now, ttl: Ttl) -> bool:
    if at is None:
        return False
    return (now - at) <= ttl.window()


def _last_stream_at(ev: TrainEvidence):
    candidates = [ev.tb_last_decode_at, ev.eval_npz_mtime]
    fresh = [t for t in candidates if t is not None]
    return max(fresh) if fresh else None


def status_train(ev: TrainEvidence, now, ttl: Ttl) -> tuple[Status, StatusReason]:
    if _has_failure(ev.log_tokens):
        return "failed", "failure_token"
    if ev.final_zip is not None:
        return "complete", "terminal_artifact"
    last = _last_stream_at(ev)
    if last is not None and _within_ttl(last, now, ttl):
        return "running", "stream_fresh"
    if ev.tb_max_step is None and ev.tb_last_decode_at is None and ev.eval_npz_mtime is None:
        if _within_ttl(ev.dir_mtime, now, ttl):
            return "waiting", "parent_waiting"
        return "stale", "ttl_expired"
    if last is not None:
        return "stale", "ttl_expired"
    return "unknown", "no_evidence"


def status_collect(ev: CollectEvidence, now, ttl: Ttl) -> tuple[Status, StatusReason]:
    if _has_failure(ev.log_tokens):
        return "failed", "failure_token"
    if (
        ev.todo_count == 0
        and ev.claimed_count == 0
        and ev.partial_npz == 0
        and ev.complete_npz > 0
    ):
        return "complete", "terminal_artifact"
    if ev.last_seed_at is not None and _within_ttl(ev.last_seed_at, now, ttl):
        return "running", "stream_fresh"
    if ev.todo_count > 0 and ev.complete_npz == 0 and _within_ttl(ev.last_seed_at or now, now, ttl):
        return "waiting", "parent_waiting"
    if ev.last_seed_at is not None:
        return "stale", "ttl_expired"
    if ev.todo_count > 0:
        return "waiting", "parent_waiting"
    return "unknown", "no_evidence"


def status_gate_c(ev: GateCEvidence, now, ttl: Ttl) -> tuple[Status, StatusReason]:
    if _has_failure(ev.log_tokens):
        return "failed", "failure_token"
    has_train_ok = "TRAIN_OK" in ev.log_tokens
    has_eval_ok = "EVAL_OK" in ev.log_tokens
    if ev.ce_zip is not None and has_train_ok and has_eval_ok:
        return "complete", "terminal_artifact"
    last = ev.last_shard_at or ev.dataset_mtime
    if last is not None and _within_ttl(last, now, ttl):
        return "running", "stream_fresh"
    if ev.shard_count > 0 or ev.has_dataset:
        if last is not None:
            return "stale", "ttl_expired"
    return "unknown", "no_evidence"


def pick_active_run(index: EvidenceIndex, now, ttl: Ttl) -> RunId | None:
    rank = {
        "running": 5,
        "waiting": 3,
        "stale": 2,
        "unknown": 1,
        "failed": 0,
        "complete": -1,
    }
    best: tuple | None = None
    for ev in index.train:
        st, _ = status_train(ev, now, ttl)
        last = _last_stream_at(ev) or ev.dir_mtime
        key = (rank.get(st, 0), last, ev.run_id)
        if best is None or key > best:
            best = key
    return best[2] if best else None


def _train_facts(ev: TrainEvidence, for_zips: bool) -> tuple[ObservedFact, ...]:
    facts: list[ObservedFact] = []
    if not for_zips:
        if ev.tb_max_step is not None:
            for scalar in ev.tb_scalars:
                facts.append(scalar)
            facts.append(
                ObservedFact(kind="tb_step", step=ev.tb_max_step, tag="rollout/ep_rew_mean")
            )
        last = _last_stream_at(ev)
        if last is not None:
            facts.append(ObservedFact(kind="last_advance_at", at=last))
        if ev.eval_npz_rows is not None:
            facts.append(ObservedFact(kind="npz_rows", n_rows=ev.eval_npz_rows))
    if ev.final_zip is not None:
        facts.append(
            ObservedFact(
                kind="zip_presence",
                path=ev.final_zip.relpath,
                at=ev.final_zip.mtime,
            )
        )
    if ev.best_zip is not None and not for_zips:
        facts.append(
            ObservedFact(
                kind="zip_presence",
                path=ev.best_zip.relpath,
                at=ev.best_zip.mtime,
            )
        )
    return tuple(facts)


def _train_for_cells(
    index: EvidenceIndex, now, ttl: Ttl, selected: RunId | None
) -> TrainEvidence | None:
    wanted = selected or pick_active_run(index, now, ttl)
    if wanted is not None:
        for ev in index.train:
            if ev.run_id == wanted:
                return ev
    return index.train[0] if index.train else None


def _cell_for_binding(
    binding,
    index: EvidenceIndex,
    now,
    ttl: Ttl,
    train: TrainEvidence | None,
) -> OverlayCell:
    if not binding.observable:
        reason: StatusReason = "static_legend" if binding.lane == "legend" else "unobservable"
        detail = binding.note
        if binding.overlay_id == "tg-rows":
            reason = "static_legend"
        return OverlayCell(
            binding.overlay_id,
            binding.dom_kind,
            "unknown",
            reason,
            False,
            (ObservedFact(kind="unobservable", detail=detail),),
        )

    if binding.overlay_id == OverlayId("train-go"):
        if train is None:
            return OverlayCell(
                binding.overlay_id,
                binding.dom_kind,
                "unknown",
                "no_evidence",
                False,
                (),
            )
        st, reason = status_train(train, now, ttl)
        return OverlayCell(
            binding.overlay_id,
            binding.dom_kind,
            st,
            reason,
            False,
            _train_facts(train, for_zips=False),
        )

    if binding.overlay_id == OverlayId("train-zips"):
        if train is None:
            return OverlayCell(
                binding.overlay_id,
                binding.dom_kind,
                "unknown",
                "no_evidence",
                False,
                (),
            )
        if train.final_zip is not None:
            st, reason = "complete", "terminal_artifact"
        else:
            st, reason = status_train(train, now, ttl)
            if st == "complete":
                st, reason = "running", "stream_fresh"
        pulse = train.final_zip is not None or train.best_zip is not None
        return OverlayCell(
            binding.overlay_id,
            binding.dom_kind,
            st,
            reason,
            pulse,
            _train_facts(train, for_zips=True),
        )

    if binding.overlay_id == OverlayId("train-to-eval"):
        if train is not None and train.eval_npz_mtime is not None:
            if _within_ttl(train.eval_npz_mtime, now, ttl):
                st, reason = "running", "stream_fresh"
            else:
                st, reason = "stale", "ttl_expired"
            facts = (ObservedFact(kind="npz_rows", n_rows=train.eval_npz_rows),)
            return OverlayCell(
                binding.overlay_id, binding.dom_kind, st, reason, False, facts
            )
        if index.collect:
            ev = max(index.collect, key=lambda c: (c.last_seed_at or _EPOCH, c.run_id))
            st, reason = status_collect(ev, now, ttl)
            return OverlayCell(
                binding.overlay_id, binding.dom_kind, st, reason, False, ()
            )
        return OverlayCell(
            binding.overlay_id,
            binding.dom_kind,
            "unknown",
            "no_evidence",
            False,
            (),
        )

    if binding.overlay_id == OverlayId("train-to-gate"):
        if not index.gate_c:
            return OverlayCell(
                binding.overlay_id,
                binding.dom_kind,
                "unknown",
                "no_evidence",
                False,
                (),
            )
        ev = max(
            index.gate_c,
            key=lambda g: (g.last_shard_at or g.dataset_mtime or _EPOCH, g.run_id),
        )
        st, reason = status_gate_c(ev, now, ttl)
        return OverlayCell(
            binding.overlay_id, binding.dom_kind, st, reason, False, ()
        )

    return OverlayCell(
        binding.overlay_id,
        binding.dom_kind,
        "unknown",
        "unobservable",
        False,
        (ObservedFact(kind="unobservable", detail=binding.note),),
    )


def _roster(index: EvidenceIndex, now, ttl: Ttl) -> tuple[RunRosterRow, ...]:
    rows: list[RunRosterRow] = []
    for ev in index.train:
        st, _ = status_train(ev, now, ttl)
        rows.append(
            RunRosterRow(
                ev.run_id,
                "train",
                ev.relpath,
                st,
                ev.tb_max_step,
                ev.final_zip is not None,
                _last_stream_at(ev),
            )
        )
    for ev in index.collect:
        st, _ = status_collect(ev, now, ttl)
        rows.append(
            RunRosterRow(
                ev.run_id,
                "collect",
                ev.relpath,
                st,
                None,
                False,
                ev.last_seed_at,
            )
        )
    for ev in index.gate_c:
        st, _ = status_gate_c(ev, now, ttl)
        rows.append(
            RunRosterRow(
                ev.run_id,
                "gate_c",
                ev.relpath,
                st,
                None,
                ev.ce_zip is not None,
                ev.last_shard_at,
            )
        )
    return tuple(rows)


def _metrics_train(ev: TrainEvidence) -> tuple[LabeledMetric, ...]:
    out: list[LabeledMetric] = []
    if ev.tb_max_step is not None:
        out.append(LabeledMetric("tb_step", "训练 timestep", ev.tb_max_step, "step"))
    if ev.eval_npz_rows is not None:
        out.append(LabeledMetric("eval_episodes", "评测局数", ev.eval_npz_rows, "episode"))
    return tuple(out)


def _metrics_collect(ev: CollectEvidence) -> tuple[LabeledMetric, ...]:
    return (
        LabeledMetric("seed_count", "完成种子", ev.complete_npz, "seed"),
    )


def _metrics_gate(ev: GateCEvidence) -> tuple[LabeledMetric, ...]:
    return (LabeledMetric("shard_count", "shard 数", ev.shard_count, "shard"),)


def _train_occupant(ev: TrainEvidence, now, ttl: Ttl) -> Occupant:
    st, reason = status_train(ev, now, ttl)
    if st == "waiting":
        oid, sid, lab = OverlayId("train-cmd"), "launch", "Launch"
    elif st == "complete":
        oid, sid, lab = OverlayId("train-zips"), "zips", "Checkpoints"
    else:
        oid, sid, lab = OverlayId("train-go"), "learn", "learn()"
    return Occupant(
        ev.run_id,
        "train",
        oid,
        sid,
        lab,
        st,
        reason,
        _metrics_train(ev),
        _last_stream_at(ev),
    )


def _occupants(index: EvidenceIndex, now, ttl: Ttl) -> tuple[Occupant, ...]:
    rows: list[Occupant] = []
    for ev in index.train:
        rows.append(_train_occupant(ev, now, ttl))
    for ev in index.collect:
        st, reason = status_collect(ev, now, ttl)
        rows.append(
            Occupant(
                ev.run_id,
                "collect",
                OverlayId("train-to-eval"),
                "seed",
                "seed npz",
                st,
                reason,
                _metrics_collect(ev),
                ev.last_seed_at,
            )
        )
    for ev in index.gate_c:
        st, reason = status_gate_c(ev, now, ttl)
        rows.append(
            Occupant(
                ev.run_id,
                "gate_c",
                OverlayId("train-to-gate"),
                "shard",
                "lab gates",
                st,
                reason,
                _metrics_gate(ev),
                ev.last_shard_at,
            )
        )
    return tuple(rows)


def _activity_edges(train: TrainEvidence | None, now, ttl: Ttl) -> tuple[ActivityEdge, ...]:
    src, dst = OverlayId("train-go"), OverlayId("train-zips")
    if train is None:
        return (ActivityEdge(src, dst, False, "no_train_evidence"),)
    st, reason = status_train(train, now, ttl)
    if st == "running":
        return (ActivityEdge(src, dst, True, None),)
    missing = {
        "waiting": "waiting_no_stream",
        "complete": "run_complete",
        "failed": "failure_token",
        "stale": "ttl_expired",
        "unknown": "no_evidence",
    }.get(st, reason)
    return (ActivityEdge(src, dst, False, missing),)


def fold(
    index: EvidenceIndex, cfg: MonitorConfig, now, selected_run_id: RunId | None = None
) -> Snapshot:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ttl = cfg.ttl
    train = _train_for_cells(index, now, ttl, selected_run_id)
    cells = tuple(_cell_for_binding(b, index, now, ttl, train) for b in overlay_bindings())
    return Snapshot(
        schema=1,
        test_mode=cfg.test_mode,
        watch_root=str(cfg.watch_root),
        generated_at=now,
        ttl_seconds=ttl.seconds,
        active_run_id=selected_run_id or pick_active_run(index, now, ttl),
        cells=cells,
        runs=_roster(index, now, ttl),
        occupants=_occupants(index, now, ttl),
        activity_edges=_activity_edges(train, now, ttl),
        legend=static_legend(),
        recent_deliveries=(),
        declared_timesteps=cfg.declared_timesteps,
    )


def diff_status(prev: Snapshot, nxt: Snapshot, now) -> tuple[StatusChanged, ...]:
    prev_map = {c.overlay_id: c for c in prev.cells}
    out: list[StatusChanged] = []
    for cell in nxt.cells:
        old = prev_map.get(cell.overlay_id)
        if old is None or old.status != cell.status or old.reason != cell.reason:
            out.append(
                StatusChanged(
                    type="status_changed",
                    overlay_id=cell.overlay_id,
                    status=cell.status,
                    reason=cell.reason,
                    observed_at=now,
                )
            )
    return tuple(out)
