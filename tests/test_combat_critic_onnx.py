"""Colab critic ONNX export / ORT verify (no Drive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from sts2_env.colab.combat_critic import (
    CriticTrainConfig,
    critic_onnx_verify_tolerance,
    export_critic_onnx,
    train_critic_smoke,
    verify_critic_onnx,
)
from tests.test_combat_feature_extract import _write_fixture_npz


def test_colab_v1_onnx_verify_combined_gate():
    y_large = np.array([[1e4], [-1e4]], dtype=np.float32)
    tol = critic_onnx_verify_tolerance(
        y_large, gate="colab_v1", max_abs_err=1e-5
    )
    assert tol == pytest.approx(0.1)
    y_small = np.array([[0.01]], dtype=np.float32)
    assert critic_onnx_verify_tolerance(
        y_small, gate="colab_v1", max_abs_err=1e-5
    ) == pytest.approx(1e-5)
    assert critic_onnx_verify_tolerance(
        y_small, gate="absolute", max_abs_err=1e-4
    ) == pytest.approx(1e-4)


def test_export_and_verify_critic_onnx(tmp_path: Path):
    src = tmp_path / "transitions.npz"
    _write_fixture_npz(src, n=1200)
    ckpt = tmp_path / "critic_smoke.pt"
    train_critic_smoke(
        src,
        ckpt,
        config=CriticTrainConfig(epochs=1, min_rows=1000, batch_size=128, seed=0),
    )
    onnx = tmp_path / "critic_smoke.onnx"
    out = export_critic_onnx(ckpt, onnx)
    assert onnx.is_file()
    assert out == str(onnx.resolve())
    meta = verify_critic_onnx(ckpt, onnx, batch_size=16, seed=1)
    assert meta["max_abs_err"] <= 1e-4
    assert np.isfinite(meta["max_abs_err"])
