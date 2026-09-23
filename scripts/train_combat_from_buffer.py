#!/usr/bin/env python3
"""Fine-tune bh_v1 from a collected hang-protocol combat buffer.

``MaskablePPO.learn`` steps ``CombatReplayEnv`` (and loadout-dominant
``loadout_v1`` mix). TypeSafe/Jev is not on this path — collect first with
``scripts/collect_runenv_combat.py``.

Usage:
    python scripts/train_combat_from_buffer.py --dry-run
    python scripts/train_combat_from_buffer.py \\
        --buffer output/runenv_combat_buffer/transitions.npz \\
        --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \\
        --output-dir output/combat_runenv_offline_ld03 \\
        --runenv-frac 0.3 --device auto --n-envs 1 --total-timesteps 2048

Never overwrites ``bh_v1``, ``combat_runenv_onpolicy_v1``, or
``combat_runenv_antiforget_v1``. Never continue-from the antiforget_v1 zip.
Hang eval flags/bars unchanged. Not a win claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sts2_env.eval.combat_hold import HOLD_BOSS_MIN, HOLD_OVERALL_MIN, run_hold_smoke
from sts2_env.gym_env.combat_buffer import (
    FROZEN_OUTDIR_NAMES,
    CombatReplayEnv,
    hang_protocol_meta,
    load_combat_buffer,
    refuse_frozen_path,
    synthetic_combat_buffer,
)
from sts2_env.gym_env.runenv_antiforget import (
    DEFAULT_RUNENV_FRAC,
    MaskedEnvMaker,
    MixedHangLoadoutEnvMaker,
    frozen_runenv_frac_warning,
    parse_runenv_frac,
    refuse_antiforget_v1_continue,
)
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    hang_jev_flags,
)

HUNG_OUTDIR_NAME = "combat_ppo_obs_v1_bh_v1"
HUNG_COMBAT_ZIP = f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
DEFAULT_OUTPUT_DIR = "output/combat_runenv_offline_ld03"
PROTOCOL_ID = "hang_protocol_runenv_offline_buffer_ld03 LOCKED 2026-09-23"


def refuse_overwrite_frozen(output_dir: str | Path) -> Path:
    return refuse_frozen_path(output_dir, what="outdir")


def resolve_continue_from(path: str, *, must_exist: bool = True) -> Path:
    zip_path = refuse_antiforget_v1_continue(path)
    if must_exist and not zip_path.is_file():
        raise SystemExit(f"continue-from zip not found: {path}")
    return zip_path


def model_obs_dim(model: Any) -> int:
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def require_combat_obs_dim(model: Any, expected: int | None = None) -> int:
    from sts2_env.gym_env.observation import OBS_SIZE

    expected = OBS_SIZE if expected is None else int(expected)
    obs_dim = model_obs_dim(model)
    if obs_dim != expected:
        raise SystemExit(
            f"continue-from obs_dim={obs_dim} != combat OBS_SIZE={expected}"
        )
    return obs_dim


def require_train_deps():
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise SystemExit(
            "Training requires sb3-contrib and stable-baselines3. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    return MaskablePPO, ActionMasker, DummyVecEnv, SubprocVecEnv, BaseCallback


def make_masked_env(
    seed: int,
    *,
    buffer_path: str | None,
    runenv_frac: float,
    max_steps: int = 2000,
    ActionMasker: Any = None,
):
    inner = MixedHangLoadoutEnvMaker(
        seed=seed,
        runenv_frac=runenv_frac,
        max_steps=max_steps,
        buffer_path=buffer_path,
    )
    return MaskedEnvMaker(inner, ActionMasker=ActionMasker)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune bh_v1 on collected hang combat segments "
            "(loadout-dominant mix). Jev is not on the learn path. "
            "0.7 antiforget_v1 is frozen."
        )
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default="",
        help="transitions.npz from collect_runenv_combat.py (optional for --dry-run)",
    )
    parser.add_argument(
        "--continue-from",
        "--model",
        dest="continue_from",
        default=HUNG_COMBAT_ZIP,
        help=f"Combat MaskablePPO zip (obs_v1=181). Default: {HUNG_COMBAT_ZIP}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"New outdir (not {HUNG_OUTDIR_NAME} / combat_runenv_onpolicy_v1 "
            "/ combat_runenv_antiforget_v1)"
        ),
    )
    parser.add_argument(
        "--runenv-frac",
        type=float,
        default=DEFAULT_RUNENV_FRAC,
        help=(
            "P(buffer/RunEnv episode); rest live loadout_v1 "
            "(default 0.3 loadout-dominant; 0.7 frozen)"
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2048,
        help="Combat PPO timesteps (default 2048 CPU smoke; not a 500k default)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Parallel replay/mix envs. >1 uses SubprocVecEnv (no Jev on replay half)",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--hold-freq",
        type=int,
        default=0,
        help="Every N timesteps run loadout_v1 HOLD smoke (0=skip)",
    )
    parser.add_argument("--hold-n-eps", type=int, default=1)
    parser.add_argument("--hold-stop", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay env (+ mix) from buffer or synthetic; no SB3 learn",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SB3 device: auto (cuda if available else cpu), cuda, cpu, or cuda:N",
    )
    args = parser.parse_args(argv)
    args.runenv_frac = parse_runenv_frac(args.runenv_frac)
    return args


def resolve_device(spec: str, *, cuda_available: bool | None = None) -> str:
    """Map --device to an SB3 device string. EP 3070 Ti: auto or cuda."""
    s = (spec or "auto").strip().lower()
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
    if s in ("auto", "cuda-if-available", ""):
        return "cuda" if cuda_available else "cpu"
    if s == "cuda":
        if not cuda_available:
            raise SystemExit(
                "--device cuda requested but torch.cuda.is_available() is False"
            )
        return "cuda"
    if s == "cpu":
        return "cpu"
    if s.startswith("cuda:"):
        if not cuda_available:
            raise SystemExit(f"--device {s} requested but CUDA unavailable")
        return s
    raise SystemExit(f"unknown --device {spec!r} (use auto|cuda|cpu|cuda:N)")


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    from sts2_env.core.constants import ACTION_SPACE_SIZE
    from sts2_env.eval.combat_hold import hold_jobs
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.runenv_antiforget import (
        MixedHangLoadoutEnv,
        make_loadout_v1_provider,
    )

    out = refuse_overwrite_frozen(args.output_dir)
    refuse_antiforget_v1_continue(args.continue_from)
    flags = hang_jev_flags()
    buffer_path = args.buffer.strip()
    if buffer_path:
        arrays, meta = load_combat_buffer(buffer_path)
        replay = CombatReplayEnv(arrays, meta, seed=0)
        source = "disk"
    else:
        arrays = synthetic_combat_buffer(n=16, n_episodes=4, seed=0)
        meta = hang_protocol_meta()
        replay = CombatReplayEnv(arrays, meta, seed=0)
        source = "synthetic"
    obs_b, info_b = replay.reset(seed=0)
    mask_b = replay.action_masks()
    local = int(np.flatnonzero(mask_b == 1)[0])
    obs_b2, reward_b, _term, _trunc, info_b2 = replay.step(local)
    mix = MixedHangLoadoutEnv(
        runenv_frac=1.0,
        loadout_provider=make_loadout_v1_provider(offset=0),
        max_steps=40,
        runenv_env=replay,
    )
    obs_m, info_m = mix.reset(seed=1)
    mix.close()
    env_l = MixedHangLoadoutEnv(
        runenv_frac=0.0,
        loadout_provider=make_loadout_v1_provider(offset=1),
        max_steps=40,
    )
    obs_l, info_l = env_l.reset(seed=1)
    env_l.close()
    jobs = hold_jobs(n_eps=1)
    return {
        "protocol": PROTOCOL_ID,
        "dry_run": True,
        "continue_from": args.continue_from,
        "output_dir": str(out),
        "buffer": buffer_path or None,
        "buffer_source": source,
        "n_transitions": int(arrays["obs"].shape[0]),
        "runenv_frac": args.runenv_frac,
        "loadout_frac": round(1.0 - args.runenv_frac, 4),
        "runenv_frac_warning": frozen_runenv_frac_warning(args.runenv_frac),
        "recipe": "loadout_dominant_0.3",
        "continue_from_policy": "bh_v1_only",
        "loadout_half": "loadout_v1",
        "n_envs": args.n_envs,
        "n_envs_backend": "SubprocVecEnv" if int(args.n_envs) > 1 else "DummyVecEnv",
        "subproc_maker": "MixedHangLoadoutEnvMaker",
        "total_timesteps": args.total_timesteps,
        "jev_on_learn_path": False,
        "device_requested": args.device,
        "device": resolve_device(args.device),
        "hang": {
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "start_with_neow": HANG_START_WITH_NEOW,
            "allows_event": flags.allows_event(),
            "allows_neow": flags.allows_neow(),
        },
        "buffer_reset": {
            "obs_size": int(obs_b.shape[-1]),
            "mask_size": int(mask_b.shape[-1]),
            "replay": info_b.get("replay"),
            "reward": float(reward_b),
            "next_obs_size": int(obs_b2.shape[-1]),
            "jev_event": info_b.get("jev_event"),
            "jev_neow": info_b.get("jev_neow"),
        },
        "mix_buffer_reset": {
            "mix_source": info_m.get("mix_source"),
            "replay": info_m.get("replay"),
            "obs_size": int(obs_m.shape[-1]),
        },
        "loadout_reset": {
            "mix_source": info_l.get("mix_source"),
            "obs_size": int(obs_l.shape[-1]),
            "loadout": info_l.get("loadout"),
        },
        "hold": {
            "overall_min": HOLD_OVERALL_MIN,
            "boss_min": HOLD_BOSS_MIN,
            "jobs_n_eps1": len(jobs),
        },
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SPACE_SIZE,
        "frozen_outdirs": list(FROZEN_OUTDIR_NAMES),
        "online_antiforget_still_valid": True,
        "antiforget_v1_0.7": "frozen",
    }


def make_hold_callback(
    BaseCallback: Any,
    *,
    output_dir: Path,
    hold_freq: int,
    n_eps: int,
    stop_on_fail: bool,
):
    class HoldCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last_ts = -1
            self.best_score = -1.0
            self.history: list[dict[str, Any]] = []

        def _on_step(self) -> bool:
            if hold_freq <= 0:
                return True
            ts = int(self.num_timesteps)
            if ts == 0 or ts % hold_freq != 0 or ts == self._last_ts:
                return True
            self._last_ts = ts
            model = self.model

            def predict_fn(obs, mask):
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                return int(action)

            summary = run_hold_smoke(predict_fn, n_eps=n_eps)
            overall = float(summary["overall"]["win_rate"])
            boss = float(summary["boss"]["win_rate"])
            passed = bool(summary["passed"])
            row = {
                "timesteps": ts,
                "overall": overall,
                "elite": float(summary["elite"]["win_rate"]),
                "boss": boss,
                "passed": passed,
            }
            self.history.append(row)
            (output_dir / "hold_logs.json").write_text(
                json.dumps(self.history, indent=2) + "\n"
            )
            print(
                f"HOLD smoke t={ts} overall={overall:.1%} "
                f"elite={row['elite']:.1%} boss={boss:.1%} "
                f"{'PASS' if passed else 'FAIL'} "
                f"(gate {HOLD_OVERALL_MIN:.0%}/{HOLD_BOSS_MIN:.0%})"
            )
            score = overall + 0.01 * boss
            if passed and score >= self.best_score:
                self.best_score = score
                dest = output_dir / "best_hold" / "best_model"
                dest.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(dest))
            if stop_on_fail and not passed:
                return False
            return True

    return HoldCheckpointCallback()


def train(args: argparse.Namespace) -> None:
    buffer_path = args.buffer.strip()
    if not buffer_path:
        raise SystemExit("--buffer is required unless --dry-run")
    MaskablePPO, ActionMasker, DummyVecEnv, SubprocVecEnv, BaseCallback = (
        require_train_deps()
    )
    from sts2_env.gym_env.observation import OBS_SIZE

    output_dir = refuse_overwrite_frozen(args.output_dir)
    continue_from = resolve_continue_from(args.continue_from, must_exist=True)
    arrays, _meta = load_combat_buffer(buffer_path)
    flags = hang_jev_flags()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, int(args.n_envs))
    device = resolve_device(getattr(args, "device", "auto"))
    print("Training MaskablePPO from hang combat buffer")
    print("  continue_from:   ", continue_from)
    print("  output_dir:      ", output_dir)
    print("  buffer:          ", buffer_path, "n=", arrays["obs"].shape[0])
    print("  runenv_frac:     ", args.runenv_frac)
    note = frozen_runenv_frac_warning(args.runenv_frac)
    if note:
        print(" ", note)
    print("  n_envs:          ", n_envs, "(SubprocVecEnv)" if n_envs > 1 else "(DummyVecEnv)")
    print("  device:          ", device)
    print("  total_timesteps: ", args.total_timesteps)
    print("  jev_on_learn:    ", False)
    print("  hang collect was: jev on / event off / neow off / start_with_neow")
    print("  allows_event:    ", flags.allows_event())
    print("  allows_neow:     ", flags.allows_neow())
    print()
    makers = [
        make_masked_env(
            i,
            buffer_path=buffer_path,
            runenv_frac=args.runenv_frac,
            max_steps=args.max_steps,
            ActionMasker=ActionMasker,
        )
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(makers) if n_envs > 1 else DummyVecEnv(makers)
    model = MaskablePPO.load(str(continue_from), env=train_env, device=device)
    require_combat_obs_dim(model, OBS_SIZE)

    callback = None
    if int(args.hold_freq) > 0:
        callback = make_hold_callback(
            BaseCallback,
            output_dir=output_dir,
            hold_freq=int(args.hold_freq),
            n_eps=int(args.hold_n_eps),
            stop_on_fail=bool(args.hold_stop),
        )

    start = time.perf_counter()
    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=False,
    )
    elapsed = time.perf_counter() - start
    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    train_env.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args), ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
