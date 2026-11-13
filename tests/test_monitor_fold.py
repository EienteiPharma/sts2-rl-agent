from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sts2_env.web._monitor.clock import FrozenClock
from sts2_env.web._monitor.fold import fold, status_train
from sts2_env.web._monitor.legend import overlay_bindings, static_legend
from sts2_env.web._monitor.types import (
    CollectEvidence,
    EvidenceIndex,
    GateCEvidence,
    MonitorConfig,
    ObservedFact,
    RunId,
    TrainEvidence,
    Ttl,
    WatchRoot,
)

UTC = timezone.utc
NOW = datetime(2026, 11, 12, 13, 0, 0, tzinfo=UTC)
TTL = Ttl(120.0)


def _cfg() -> MonitorConfig:
    return MonitorConfig(
        watch_root=Path("."),
        ttl=TTL,
        test_mode=False,
        preferred_port=None,
        declared_timesteps=99_999_999,
    )


def _train(**kwargs) -> TrainEvidence:
    base = dict(
        run_id=RunId("run_a"),
        relpath="run_a",
        dir_mtime=NOW - timedelta(seconds=10),
        tb_max_step=None,
        tb_last_decode_at=None,
        tb_scalars=(),
        eval_npz_rows=None,
        eval_npz_mtime=None,
        best_zip=None,
        final_zip=None,
        log_tokens=(),
    )
    base.update(kwargs)
    return TrainEvidence(**base)


def test_waiting_no_tb_fresh_dir_mtime():
    ev = _train(dir_mtime=NOW - timedelta(seconds=30))
    st, reason = status_train(ev, NOW, TTL)
    assert st == "waiting"
    assert reason == "parent_waiting"
    snap = fold(EvidenceIndex((ev,), (), ()), _cfg(), NOW)
    assert snap.cell("train-go").status == "waiting"


def test_running_tb_inside_ttl():
    ev = _train(
        tb_max_step=50_000_000,
        tb_last_decode_at=NOW - timedelta(seconds=30),
        tb_scalars=(
            ObservedFact(kind="tb_scalar", tag="eval/mean_reward", step=100, value=-0.5),
        ),
    )
    snap = fold(EvidenceIndex((ev,), (), ()), _cfg(), NOW)
    assert snap.cell("train-go").status == "running"
    facts = snap.cell("train-go").facts
    assert any(f.kind == "tb_scalar" and f.tag == "eval/mean_reward" for f in facts)
    assert snap.cell("te-run").status == "unknown"


def test_stale_not_failed():
    ev = _train(
        tb_max_step=1,
        tb_last_decode_at=NOW - timedelta(seconds=TTL.seconds + 1),
    )
    st, _ = status_train(ev, NOW, TTL)
    assert st == "stale"
    snap = fold(EvidenceIndex((ev,), (), ()), _cfg(), NOW)
    assert snap.cell("train-go").status == "stale"
    assert snap.cell("train-zips").status != "failed"


def test_cells_follow_active_running_not_complete_first():
    from sts2_env.web._monitor.types import ZipPresence

    complete = _train(
        run_id=RunId("a_complete"),
        relpath="a_complete",
        final_zip=ZipPresence("final_model.zip", NOW - timedelta(seconds=5), 100),
    )
    running = _train(
        run_id=RunId("z_running"),
        relpath="z_running",
        tb_max_step=7,
        tb_last_decode_at=NOW - timedelta(seconds=10),
    )
    snap = fold(EvidenceIndex((complete, running), (), ()), _cfg(), NOW)
    assert snap.active_run_id == "z_running"
    assert snap.cell("train-go").status == "running"
    assert len(snap.runs) == 2


def test_fold_selected_run_id():
    from sts2_env.web._monitor.types import ZipPresence

    complete = _train(
        run_id=RunId("done"),
        relpath="done",
        final_zip=ZipPresence("final_model.zip", NOW - timedelta(seconds=5), 100),
    )
    running = _train(
        run_id=RunId("live"),
        relpath="live",
        tb_max_step=3,
        tb_last_decode_at=NOW - timedelta(seconds=10),
    )
    snap = fold(
        EvidenceIndex((complete, running), (), ()),
        _cfg(),
        NOW,
        selected_run_id=RunId("done"),
    )
    assert snap.active_run_id == "done"
    assert snap.cell("train-zips").status == "complete"


def test_final_zip_complete():
    from sts2_env.web._monitor.types import ZipPresence

    ev = _train(
        final_zip=ZipPresence("final_model.zip", NOW - timedelta(seconds=5), 100),
    )
    snap = fold(EvidenceIndex((ev,), (), ()), _cfg(), NOW)
    assert snap.cell("train-zips").status == "complete"
    assert snap.cell("te-run").status == "unknown"


def test_worker_fail_failed_missing_zip_not_failed():
    ev_fail = _train(log_tokens=("WORKER_FAIL",))
    snap_fail = fold(EvidenceIndex((ev_fail,), (), ()), _cfg(), NOW)
    assert snap_fail.cell("train-go").status == "failed"
    ev_miss = _train()
    snap_miss = fold(EvidenceIndex((ev_miss,), (), ()), _cfg(), NOW)
    assert snap_miss.cell("train-go").status != "failed"


def test_tg_rows_unknown_legend_not_verdict():
    snap = fold(EvidenceIndex((), (), ()), _cfg(), NOW)
    cell = snap.cell("tg-rows")
    assert cell.status == "unknown"
    assert cell.reason == "static_legend"
    assert all(item.not_a_verdict for item in static_legend().items)


def test_declared_timesteps_not_used_in_status():
    ev = _train(dir_mtime=NOW - timedelta(seconds=30))
    cfg = _cfg()
    snap = fold(EvidenceIndex((ev,), (), ()), cfg, NOW)
    assert snap.declared_timesteps == 99_999_999
    assert snap.cell("train-go").status == "waiting"


def test_watch_root_rejects_escape(tmp_path):
    root = tmp_path / "watch"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    wr = WatchRoot(root)
    with pytest.raises(ValueError):
        wr.rel(outside)


def test_overlay_bindings_ids_nonempty():
    bindings = overlay_bindings()
    assert {b.overlay_id for b in bindings} >= {
        "train-go",
        "train-zips",
        "te-run",
        "tg-rows",
    }


def test_frozen_clock_parse():
    clk = FrozenClock.parse("2026-11-12T13:00:00+00:00")
    assert clk.now() == NOW


def test_activity_edge_running_stops_when_stale():
    running = _train(
        tb_max_step=9,
        tb_last_decode_at=NOW - timedelta(seconds=10),
    )
    snap = fold(EvidenceIndex((running,), (), ()), _cfg(), NOW)
    ae = snap.activity_edges[0]
    assert ae.edge_from == "train-go"
    assert ae.edge_to == "train-zips"
    assert ae.active is True
    later = NOW + timedelta(seconds=TTL.seconds + 5)
    snap2 = fold(EvidenceIndex((running,), (), ()), _cfg(), later)
    assert snap2.activity_edges[0].active is False
    assert snap2.activity_edges[0].missing_signal == "ttl_expired"


def test_pick_active_prefers_running_over_newer_waiting():
    waiting = _train(run_id=RunId("w"), relpath="w", dir_mtime=NOW - timedelta(seconds=1))
    running = _train(
        run_id=RunId("r"),
        relpath="r",
        dir_mtime=NOW - timedelta(seconds=40),
        tb_max_step=12,
        tb_last_decode_at=NOW - timedelta(seconds=10),
    )
    snap = fold(EvidenceIndex((waiting, running), (), ()), _cfg(), NOW)
    assert snap.active_run_id == "r"


def test_occupants_map_inner_nodes_not_root():
    from sts2_env.web._monitor.types import ZipPresence

    waiting = _train(run_id=RunId("w"), relpath="w")
    running = _train(
        run_id=RunId("r"),
        relpath="r",
        tb_max_step=12,
        tb_last_decode_at=NOW - timedelta(seconds=10),
        eval_npz_rows=4,
    )
    done = _train(
        run_id=RunId("d"),
        relpath="d",
        final_zip=ZipPresence("final_model.zip", NOW - timedelta(seconds=5), 100),
    )
    collect = CollectEvidence(
        run_id=RunId("c1"),
        relpath="collect",
        todo_count=1,
        claimed_count=0,
        complete_npz=2,
        partial_npz=0,
        last_seed_at=NOW - timedelta(seconds=5),
        log_tokens=(),
    )
    snap = fold(EvidenceIndex((waiting, running, done), (collect,), ()), _cfg(), NOW)
    by_id = {o.run_id: o for o in snap.occupants}
    assert by_id["w"].overlay_id == "train-cmd"
    assert by_id["w"].substep_label == "Launch"
    assert by_id["r"].overlay_id == "train-go"
    assert by_id["r"].substep_label == "learn()"
    assert by_id["d"].overlay_id == "train-zips"
    assert by_id["c1"].lane == "collect"
    assert by_id["c1"].overlay_id == "train-to-eval"
    assert {o.run_id for o in snap.occupants} == {"w", "r", "d", "c1"}
    train_m = {m.kind: m for m in by_id["r"].metrics}
    assert train_m["tb_step"].label == "训练 timestep"
    assert train_m["tb_step"].total is None
    assert train_m["eval_episodes"].label == "评测局数"
    assert all(m.total is None for m in by_id["r"].metrics)


def test_collect_complete_rule():
    from sts2_env.web._monitor.fold import status_collect

    ev = CollectEvidence(
        run_id=RunId("c1"),
        relpath="collect",
        todo_count=0,
        claimed_count=0,
        complete_npz=3,
        partial_npz=0,
        last_seed_at=NOW,
        log_tokens=(),
    )
    st, _ = status_collect(ev, NOW, TTL)
    assert st == "complete"


def test_eval_cell_follows_eval_npz():
    running = _train(
        tb_max_step=12,
        tb_last_decode_at=NOW - timedelta(seconds=10),
        eval_npz_rows=4,
        eval_npz_mtime=NOW - timedelta(seconds=10),
    )
    snap = fold(EvidenceIndex((running,), (), ()), _cfg(), NOW)
    cell = snap.cell("train-to-eval")
    assert cell.status == "running"
    later = NOW + timedelta(seconds=TTL.seconds + 5)
    snap2 = fold(EvidenceIndex((running,), (), ()), _cfg(), later)
    assert snap2.cell("train-to-eval").status == "stale"
    assert snap2.cell("train-to-eval").reason == "ttl_expired"


def test_eval_cell_falls_back_to_collect():
    collect = CollectEvidence(
        run_id=RunId("c1"),
        relpath="collect",
        todo_count=1,
        claimed_count=0,
        complete_npz=1,
        partial_npz=0,
        last_seed_at=NOW - timedelta(seconds=5),
        log_tokens=(),
    )
    snap = fold(EvidenceIndex((), (collect,), ()), _cfg(), NOW)
    assert snap.cell("train-to-eval").status == "running"
    assert snap.occupants[0].overlay_id == "train-to-eval"


def test_gate_cell_and_occupant_use_train_to_gate():
    gate = GateCEvidence(
        run_id=RunId("g1"),
        relpath="gate_running",
        shard_count=1,
        last_shard_at=NOW - timedelta(seconds=5),
        has_dataset=False,
        dataset_mtime=None,
        ce_zip=None,
        log_tokens=(),
    )
    snap = fold(EvidenceIndex((), (), (gate,)), _cfg(), NOW)
    assert snap.cell("train-to-gate").status == "running"
    assert snap.occupants[0].overlay_id == "train-to-gate"
    assert snap.occupants[0].substep_label == "lab gates"
    later = NOW + timedelta(seconds=TTL.seconds + 5)
    snap2 = fold(EvidenceIndex((), (), (gate,)), _cfg(), later)
    assert snap2.cell("train-to-gate").status == "stale"
