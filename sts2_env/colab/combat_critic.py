"""Minimal combat value / Critic skeleton for Colab (tip #2).

Input: ``obs`` vector dim 181 (combat obs_v1 from tip #1 / ``transitions.npz``).
Target: scalar **Monte-Carlo return** from ``reward`` + ``done`` (episodes split
at ``done=True``; backward accumulate ``G_t = r_t + γ G_{t+1}``, reset after terminal).

Optional **win proxy** (documented, not default): at terminal step,
``1.0`` if episode return sum > 0 else ``0.0``.

No game runtime, EP bridge, or simulator — table I/O only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sts2_env.colab.combat_transition_features import (
    COMBAT_OBS_DIM,
    load_transitions_npz,
    subsample_transitions,
    validate_transitions,
)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional until train
    torch = None  # type: ignore
    nn = None  # type: ignore

DEFAULT_GAMMA = 0.99
DEFAULT_MIN_ROWS = 1000
LABEL_MC_RETURN = "mc_return_gamma099"
LABEL_WIN_PROXY = "episode_win_proxy"


def mc_return_targets(
    reward: np.ndarray,
    done: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Backward MC return; episode boundaries where ``done`` is True."""
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.bool_).reshape(-1)
    if r.shape[0] != d.shape[0]:
        raise ValueError("reward/done length mismatch")
    n = r.size
    out = np.zeros(n, dtype=np.float32)
    g = 0.0
    for t in range(n - 1, -1, -1):
        g = float(r[t]) + float(gamma) * g
        out[t] = np.float32(g)
        if d[t]:
            g = 0.0
    return out


def episode_win_proxy_targets(
    reward: np.ndarray,
    done: np.ndarray,
) -> np.ndarray:
    """Terminal-step win proxy: 1 if episode sum(reward) > 0 else 0 (others 0)."""
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.bool_).reshape(-1)
    n = r.size
    out = np.zeros(n, dtype=np.float32)
    ep_ret = 0.0
    ep_indices: list[int] = []
    for t in range(n):
        ep_ret += float(r[t])
        ep_indices.append(t)
        if d[t]:
            win = 1.0 if ep_ret > 0.0 else 0.0
            if ep_indices:
                out[ep_indices[-1]] = np.float32(win)
            ep_ret = 0.0
            ep_indices = []
    return out


def load_obs_reward_done(
    path: str | Path,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
    sample_rows: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load tip #1 export (.npz) or raw ``transitions.npz``."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            if "next_obs" in keys and "action_mask" in keys:
                data = load_transitions_npz(p)
                validate_transitions(data)
                n = int(data["obs"].shape[0])
                take = int(sample_rows or max(min_rows, n))
                take = min(take, n)
                if n < min_rows:
                    raise ValueError(f"need >={min_rows} rows, got {n}")
                data = subsample_transitions(data, n_rows=take, seed=seed)
                validate_transitions(data)
                obs = np.asarray(data["obs"], dtype=np.float32)
                reward = np.asarray(data["reward"], dtype=np.float32)
                done = np.asarray(data["done"], dtype=np.bool_)
            else:
                obs = np.asarray(z["obs"], dtype=np.float32)
                reward = np.asarray(z["reward"], dtype=np.float32)
                done = np.asarray(z["done"], dtype=np.bool_)
                if obs.shape[0] < min_rows:
                    raise ValueError(f"need >={min_rows} rows, got {obs.shape[0]}")
    elif p.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
        cols = [f"obs_{i}" for i in range(COMBAT_OBS_DIM)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"parquet missing {missing[:3]}...")
        obs = df[cols].to_numpy(dtype=np.float32)
        reward = df["reward"].to_numpy(dtype=np.float32)
        done = df["done"].to_numpy(dtype=np.bool_)
        if len(df) < min_rows:
            raise ValueError(f"need >={min_rows} rows, got {len(df)}")
    else:
        raise ValueError(f"unsupported feature path {p}")
    if obs.shape[1] != COMBAT_OBS_DIM:
        raise ValueError(f"obs dim {obs.shape[1]} != {COMBAT_OBS_DIM}")
    if not np.isfinite(obs).all() or not np.isfinite(reward).all():
        raise ValueError("obs/reward contains NaN or Inf")
    return obs, reward, done, np.arange(obs.shape[0], dtype=np.int64)


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise ImportError("torch required for critic train; pip install torch")
    return torch


@dataclass
class CriticTrainConfig:
    gamma: float = DEFAULT_GAMMA
    label: Literal["mc_return", "win_proxy"] = "mc_return"
    hidden: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 3
    seed: int = 0
    min_rows: int = DEFAULT_MIN_ROWS


class CombatCriticMLP:
    """Thin wrapper; builds ``nn.Module`` on first use."""

    def __init__(self, obs_dim: int = COMBAT_OBS_DIM, hidden: tuple[int, ...] = (128, 64)):
        _require_torch()
        layers: list[Any] = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def parameters(self):
        return self.net.parameters()

    def train_mode(self, mode: bool = True):
        self.net.train(mode)

    def __call__(self, x):
        return self.net(x).squeeze(-1)


def build_targets(
    reward: np.ndarray,
    done: np.ndarray,
    *,
    label: Literal["mc_return", "win_proxy"] = "mc_return",
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    if label == "mc_return":
        return mc_return_targets(reward, done, gamma=gamma)
    return episode_win_proxy_targets(reward, done)


def train_critic_smoke(
    feature_path: str | Path,
    ckpt_path: str | Path,
    *,
    config: CriticTrainConfig | None = None,
) -> dict[str, Any]:
    """Smoke-train MLP critic; save ``.pt`` state + metadata."""
    t = _require_torch()
    cfg = config or CriticTrainConfig()
    t.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    obs, reward, done, _idx = load_obs_reward_done(
        feature_path,
        min_rows=cfg.min_rows,
        sample_rows=cfg.min_rows,
        seed=cfg.seed,
    )
    targets = build_targets(reward, done, label=cfg.label, gamma=cfg.gamma)
    if not np.isfinite(targets).all():
        raise ValueError("targets contain NaN or Inf")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = CombatCriticMLP(COMBAT_OBS_DIM, hidden=cfg.hidden).net.to(device)
    opt = t.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    x = t.as_tensor(obs, dtype=t.float32, device=device)
    y = t.as_tensor(targets, dtype=t.float32, device=device)
    n = x.shape[0]
    bs = min(int(cfg.batch_size), n)

    losses: list[float] = []
    for _epoch in range(int(cfg.epochs)):
        perm = t.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            pred = model(x[idx]).squeeze(-1)
            loss = loss_fn(pred, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            val = float(loss.detach().cpu().item())
            if not np.isfinite(val):
                raise ValueError(f"non-finite loss at epoch step: {val}")
            epoch_loss += val
            steps += 1
        losses.append(epoch_loss / max(steps, 1))

    ckpt = Path(ckpt_path).expanduser()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    label_name = LABEL_MC_RETURN if cfg.label == "mc_return" else LABEL_WIN_PROXY
    payload = {
        "state_dict": model.cpu().state_dict(),
        "obs_dim": COMBAT_OBS_DIM,
        "hidden": list(cfg.hidden),
        "label": label_name,
        "gamma": float(cfg.gamma) if cfg.label == "mc_return" else None,
        "loss_last": float(losses[-1]) if losses else None,
        "losses": losses,
        "n_rows": int(n),
        "source": str(Path(feature_path).resolve()),
    }
    t.save(payload, ckpt)

    return {
        "ckpt_path": str(ckpt.resolve()),
        "n_rows": int(n),
        "obs_dim": COMBAT_OBS_DIM,
        "out_dim": 1,
        "label": label_name,
        "losses": losses,
        "device": str(device),
    }


DEFAULT_ONNX_VERIFY_MAX_ABS_ERR = 1e-4


def _load_ckpt_blob(pt_path: str | Path) -> dict[str, Any]:
    t = _require_torch()
    p = Path(pt_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(p)
    try:
        blob = t.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        blob = t.load(p, map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError(f"expected tip#2 critic ckpt with state_dict: {p}")
    return blob


def load_critic_net_from_ckpt(pt_path: str | Path):
    """Rebuild tip#2 ``CombatCriticMLP`` ``.net`` and load weights (eval mode)."""
    t = _require_torch()
    blob = _load_ckpt_blob(pt_path)
    obs_dim = int(blob.get("obs_dim", COMBAT_OBS_DIM))
    hidden_raw = blob.get("hidden", (128, 64))
    hidden = tuple(int(h) for h in hidden_raw)
    model = CombatCriticMLP(obs_dim, hidden=hidden).net
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def export_critic_onnx(
    pt_path: str | Path,
    onnx_path: str | Path,
    *,
    opset_version: int = 17,
) -> str:
    """Export tip#2 critic checkpoint to ONNX (``obs`` N×181 → ``value`` N×1 float32)."""
    t = _require_torch()
    model, blob = load_critic_net_from_ckpt(pt_path)
    obs_dim = int(blob.get("obs_dim", COMBAT_OBS_DIM))
    out_p = Path(onnx_path).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    dummy = t.zeros(1, obs_dim, dtype=t.float32)
    with t.no_grad():
        t.onnx.export(
            model,
            dummy,
            str(out_p),
            input_names=["obs"],
            output_names=["value"],
            dynamic_axes={"obs": {0: "N"}, "value": {0: "N"}},
            opset_version=int(opset_version),
            dynamo=False,
        )
    return str(out_p.resolve())


def verify_critic_onnx(
    pt_path: str | Path,
    onnx_path: str | Path,
    *,
    batch_size: int = 32,
    seed: int = 0,
    max_abs_err: float = DEFAULT_ONNX_VERIFY_MAX_ABS_ERR,
) -> dict[str, Any]:
    """Compare PyTorch ``.net`` forward vs ONNX Runtime on one random batch (fp32 gate)."""
    t = _require_torch()
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError("onnxruntime required for verify; pip install onnxruntime") from exc

    model, _blob = load_critic_net_from_ckpt(pt_path)
    onnx_p = Path(onnx_path).expanduser()
    if not onnx_p.is_file():
        raise FileNotFoundError(onnx_p)

    rng = np.random.default_rng(int(seed))
    obs = rng.standard_normal((int(batch_size), COMBAT_OBS_DIM), dtype=np.float32)
    with t.no_grad():
        torch_out = model(t.as_tensor(obs)).cpu().numpy()

    sess = ort.InferenceSession(
        str(onnx_p),
        providers=["CPUExecutionProvider"],
    )
    ort_out = sess.run(["value"], {"obs": obs})[0]
    if ort_out.shape != torch_out.shape:
        raise ValueError(f"shape mismatch torch {torch_out.shape} vs onnx {ort_out.shape}")

    err = float(np.max(np.abs(torch_out - ort_out)))
    if not np.isfinite(err):
        raise ValueError("non-finite ONNX verify error")
    if err > float(max_abs_err):
        raise ValueError(
            f"ONNX max_abs_err {err} > gate {max_abs_err} (fp32 export vs ORT CPU)"
        )
    return {
        "max_abs_err": err,
        "onnx_path": str(onnx_p.resolve()),
        "pt_path": str(Path(pt_path).expanduser().resolve()),
        "batch_size": int(batch_size),
        "gate_max_abs_err": float(max_abs_err),
    }


__all__ = [
    "COMBAT_OBS_DIM",
    "CriticTrainConfig",
    "CombatCriticMLP",
    "DEFAULT_GAMMA",
    "DEFAULT_MIN_ROWS",
    "DEFAULT_ONNX_VERIFY_MAX_ABS_ERR",
    "LABEL_MC_RETURN",
    "LABEL_WIN_PROXY",
    "build_targets",
    "episode_win_proxy_targets",
    "export_critic_onnx",
    "load_critic_net_from_ckpt",
    "load_obs_reward_done",
    "mc_return_targets",
    "train_critic_smoke",
    "verify_critic_onnx",
]
