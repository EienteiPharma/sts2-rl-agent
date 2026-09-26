"""Formal Colab v1 critic train (scale helpers + tiny warm-start)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sts2_env.colab.combat_critic import (
    COLAB_V1_MIN_EPOCHS,
    COLAB_V1_MIN_SAMPLE_UPDATES,
    CriticColabV1TrainConfig,
    CriticTrainConfig,
    resolve_colab_v1_epochs,
    train_critic_colab_v1,
    train_critic_smoke,
)
from tests.test_combat_feature_extract import _write_fixture_npz


def test_resolve_colab_v1_epochs_meets_sample_updates():
    epochs, spe = resolve_colab_v1_epochs(
        475_000,
        1024,
        min_epochs=COLAB_V1_MIN_EPOCHS,
        min_sample_updates=COLAB_V1_MIN_SAMPLE_UPDATES,
    )
    assert epochs >= COLAB_V1_MIN_EPOCHS
    assert epochs * spe >= COLAB_V1_MIN_SAMPLE_UPDATES


def test_train_critic_colab_v1_warm_start(tmp_path: Path):
    src = tmp_path / "transitions.npz"
    _write_fixture_npz(src, n=4000)
    smoke = tmp_path / "smoke.pt"
    train_critic_smoke(
        src,
        smoke,
        config=CriticTrainConfig(epochs=1, min_rows=1000, batch_size=256, seed=0),
    )
    out = tmp_path / "colab_v1.pt"
    meta = train_critic_colab_v1(
        src,
        out,
        smoke,
        config=CriticColabV1TrainConfig(
            batch_size=128,
            min_rows=1000,
            min_epochs=2,
            min_sample_updates=400,
            val_frac=0.05,
        ),
    )
    assert out.is_file()
    assert meta["sample_updates"] >= 400
    assert meta["epochs"] >= 2
    assert all(np.isfinite(meta["train_losses"]))
    assert all(np.isfinite(meta["val_losses"]))
    blob = torch.load(out, map_location="cpu", weights_only=False)
    assert blob.get("train_kind") == "colab_v1_formal"
