"""Extract Colab-ready combat transition tables from ``transitions.npz``.

Pure numpy/pandas — no ``sts2_env`` gym/combat imports. Intended for GPU
Colab track #1; does not open game communication or training gates.

Expected npz keys: ``obs``, ``next_obs``, ``action``, ``reward``, ``done``,
``action_mask``. ``obs`` / ``next_obs`` shape ``(N, 181)`` float32.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

COMBAT_OBS_DIM = 181
REQUIRED_NPZ_KEYS = ("obs", "next_obs", "action", "reward", "done", "action_mask")
DEFAULT_MIN_ROWS = 1000
DEFAULT_SAMPLE_ROWS = 1000


def load_transitions_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"transitions npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        keys = set(z.files)
        missing = [k for k in REQUIRED_NPZ_KEYS if k not in keys]
        if missing:
            raise ValueError(f"npz missing keys {missing}; have {sorted(keys)}")
        return {k: np.asarray(z[k]) for k in REQUIRED_NPZ_KEYS}


def validate_transitions(data: dict[str, np.ndarray]) -> None:
    n = int(data["obs"].shape[0])
    obs = np.asarray(data["obs"], dtype=np.float32)
    nxt = np.asarray(data["next_obs"], dtype=np.float32)
    if obs.shape != (n, COMBAT_OBS_DIM):
        raise ValueError(f"obs shape {obs.shape}, expected ({n}, {COMBAT_OBS_DIM})")
    if nxt.shape != (n, COMBAT_OBS_DIM):
        raise ValueError(
            f"next_obs shape {nxt.shape}, expected ({n}, {COMBAT_OBS_DIM})"
        )
    for name in ("action", "reward", "done"):
        arr = np.asarray(data[name])
        if arr.shape[0] != n:
            raise ValueError(f"{name} length {arr.shape[0]} != N={n}")
    mask = np.asarray(data["action_mask"])
    if mask.shape[0] != n:
        raise ValueError(f"action_mask length {mask.shape[0]} != N={n}")
    for name, arr in data.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if name in ("action", "reward", "done") and np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any():
                raise ValueError(f"{name} contains NaN")


def subsample_indices(n: int, k: int, seed: int = 0) -> np.ndarray:
    k = int(min(max(k, 0), n))
    rng = np.random.default_rng(int(seed))
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False))


def subsample_transitions(
    data: dict[str, np.ndarray],
    *,
    n_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    n = int(data["obs"].shape[0])
    idx = subsample_indices(n, n_rows, seed=seed)
    return {k: np.asarray(v)[idx] for k, v in data.items()}


def transitions_to_table(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Columnar feature table for parquet/npz export (SARS + mask)."""
    obs = np.asarray(data["obs"], dtype=np.float32)
    nxt = np.asarray(data["next_obs"], dtype=np.float32)
    n = obs.shape[0]
    return {
        "obs": obs,
        "next_obs": nxt,
        "action": np.asarray(data["action"], dtype=np.int64).reshape(n),
        "reward": np.asarray(data["reward"], dtype=np.float32).reshape(n),
        "done": np.asarray(data["done"], dtype=np.bool_).reshape(n),
        "action_mask": np.asarray(data["action_mask"]),
    }


def export_feature_table(
    table: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    fmt: Literal["parquet", "npz", "auto"] = "auto",
) -> dict[str, Any]:
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    use = fmt
    if use == "auto":
        use = "parquet" if suffix == ".parquet" else "npz"
    if use == "parquet":
        import pandas as pd

        n = int(table["obs"].shape[0])
        d = int(COMBAT_OBS_DIM)
        df_dict: dict[str, Any] = {
            "action": table["action"],
            "reward": table["reward"],
            "done": table["done"],
        }
        for i in range(d):
            df_dict[f"obs_{i}"] = table["obs"][:, i]
            df_dict[f"next_obs_{i}"] = table["next_obs"][:, i]
        mask = np.asarray(table["action_mask"])
        if mask.ndim == 2:
            for j in range(mask.shape[1]):
                df_dict[f"action_mask_{j}"] = mask[:, j]
        else:
            df_dict["action_mask"] = mask.reshape(n)
        pd.DataFrame(df_dict).to_parquet(out, index=False)
    elif use == "npz":
        np.savez_compressed(out, **table)
    else:
        raise ValueError(f"unknown fmt {use!r}")
    return {
        "path": str(out.resolve()),
        "format": use,
        "n_rows": int(table["obs"].shape[0]),
        "obs_shape": tuple(table["obs"].shape),
        "next_obs_shape": tuple(table["next_obs"].shape),
        "action_mask_shape": tuple(np.asarray(table["action_mask"]).shape),
    }


def extract_combat_features(
    npz_path: str | Path,
    out_path: str | Path,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    seed: int = 0,
    fmt: Literal["parquet", "npz", "auto"] = "auto",
) -> dict[str, Any]:
    """Load npz, validate, subsample ≥ ``min_rows``, export, return summary."""
    raw = load_transitions_npz(npz_path)
    validate_transitions(raw)
    n = int(raw["obs"].shape[0])
    take = max(int(sample_rows), int(min_rows))
    take = min(take, n)
    if n < int(min_rows):
        raise ValueError(f"npz has N={n} rows; gate requires >= {min_rows}")
    subset = subsample_transitions(raw, n_rows=take, seed=seed)
    validate_transitions(subset)
    table = transitions_to_table(subset)
    meta = export_feature_table(table, out_path, fmt=fmt)
    meta["source_npz"] = str(Path(npz_path).resolve())
    meta["source_n"] = n
    meta["seed"] = int(seed)
    return meta


__all__ = [
    "COMBAT_OBS_DIM",
    "DEFAULT_MIN_ROWS",
    "DEFAULT_SAMPLE_ROWS",
    "REQUIRED_NPZ_KEYS",
    "export_feature_table",
    "extract_combat_features",
    "load_transitions_npz",
    "subsample_transitions",
    "transitions_to_table",
    "validate_transitions",
]
