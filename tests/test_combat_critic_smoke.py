"""Colab combat critic smoke (torch + numpy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sts2_env.colab.combat_critic import (
    COMBAT_OBS_DIM,
    CriticTrainConfig,
    mc_return_targets,
    train_critic_smoke,
)
from tests.test_combat_feature_extract import _write_fixture_npz


def test_mc_return_resets_at_done():
    r = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    d = np.array([False, False, True, True], dtype=np.bool_)
    g = mc_return_targets(r, d, gamma=1.0)
    assert g[1] == pytest.approx(1.0)
    assert g[3] == pytest.approx(2.0)


def test_train_critic_smoke_saves_finite_loss(tmp_path: Path):
    src = tmp_path / "transitions.npz"
    _write_fixture_npz(src, n=1200)
    ckpt = tmp_path / "critic_smoke.pt"
    meta = train_critic_smoke(
        src,
        ckpt,
        config=CriticTrainConfig(epochs=2, min_rows=1000, batch_size=128, seed=0),
    )
    assert meta["n_rows"] >= 1000
    assert meta["obs_dim"] == COMBAT_OBS_DIM
    assert meta["out_dim"] == 1
    assert ckpt.is_file()
    assert all(np.isfinite(meta["losses"]))
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt, map_location="cpu")
    assert blob["obs_dim"] == COMBAT_OBS_DIM
    assert "state_dict" in blob
