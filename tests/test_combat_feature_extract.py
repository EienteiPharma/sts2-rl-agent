"""Colab combat transition feature extract (no gym / no game I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sts2_env.colab.combat_transition_features import (
    COMBAT_OBS_DIM,
    extract_combat_features,
    load_transitions_npz,
    validate_transitions,
)


def _write_fixture_npz(path: Path, n: int = 1200) -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, COMBAT_OBS_DIM), dtype=np.float32)
    nxt = rng.standard_normal((n, COMBAT_OBS_DIM), dtype=np.float32)
    np.savez_compressed(
        path,
        obs=obs,
        next_obs=nxt,
        action=rng.integers(0, 40, size=n, dtype=np.int64),
        reward=rng.standard_normal(n, dtype=np.float32),
        done=rng.integers(0, 2, size=n, dtype=np.bool_),
        action_mask=rng.integers(0, 2, size=(n, 41), dtype=np.int8),
    )


def test_extract_parquet_meets_gate(tmp_path: Path):
    pytest.importorskip("pandas")
    src = tmp_path / "transitions.npz"
    _write_fixture_npz(src)
    out = tmp_path / "features.parquet"
    meta = extract_combat_features(src, out, min_rows=1000, sample_rows=1000, seed=1)
    assert meta["n_rows"] >= 1000
    assert meta["obs_shape"] == (1000, COMBAT_OBS_DIM)
    assert out.is_file()


def test_rejects_nan(tmp_path: Path):
    src = tmp_path / "bad.npz"
    _write_fixture_npz(src, n=100)
    data = load_transitions_npz(src)
    data["reward"][0] = np.nan
    np.savez_compressed(src, **data)
    with pytest.raises(ValueError, match="NaN"):
        validate_transitions(load_transitions_npz(src))


def test_extract_npz_roundtrip(tmp_path: Path):
    src = tmp_path / "transitions.npz"
    _write_fixture_npz(src, n=1500)
    out = tmp_path / "features.npz"
    meta = extract_combat_features(src, out, sample_rows=1100)
    assert meta["format"] == "npz"
    with np.load(out) as z:
        assert z["obs"].shape == (1100, COMBAT_OBS_DIM)
