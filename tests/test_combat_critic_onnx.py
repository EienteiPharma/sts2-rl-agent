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
    export_critic_onnx,
    train_critic_smoke,
    verify_critic_onnx,
)
from tests.test_combat_feature_extract import _write_fixture_npz


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
