from __future__ import annotations

from sts2_env.web._monitor.legend import overlay_bindings


CANVAS_IDS = {
    "train-go",
    "train-to-combat",
    "train-zips",
    "train-to-eval",
    "train-to-gate",
    "te-run",
    "tg-rows",
}


def test_overlay_ids_are_pruned_set():
    ids = {b.overlay_id for b in overlay_bindings()}
    assert ids == CANVAS_IDS


def test_eval_and_gate_bindings_are_observable():
    by_id = {b.overlay_id: b for b in overlay_bindings()}
    assert by_id["train-to-eval"].observable is True
    assert by_id["train-to-gate"].observable is True
    assert by_id["tg-rows"].observable is False


def test_overlay_js_stops_flow_on_disconnect_and_reduced():
    from pathlib import Path

    src = Path("sts2_env/web/static/train_monitor_overlay.js").read_text(encoding="utf-8")
    assert "connected && ae && ae.active && !reduced" in src
    assert "if (!connected) return false;" in src
    assert "if (reduced || !delivery.edge_from || !delivery.edge_to) return false;" in src
    assert "未流动 · 连接断开" in src
    assert "未流动 · prefers-reduced-motion" in src
