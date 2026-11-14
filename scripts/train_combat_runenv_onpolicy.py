#!/usr/bin/env python3
"""Fine-tune hung combat MaskablePPO on hang-protocol RunEnv combat segments.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat_runenv_onpolicy.py --dry-run
    python scripts/train_combat_runenv_onpolicy.py \\
        --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \\
        --output-dir output/combat_runenv_onpolicy_smoke \\
        --n-envs 1 --n-steps 64 --total-timesteps 256

Forbidden: loadout fixtures (``loadout_v*``, ``mix_neow_v1``, ``train_combat.py``).
Not a full-RunEnv PPO (``train_full_run.py``).  Never overwrites ``bh_v1``.

Hang protocol is locked in the env (see docs/RUNENV_ONPOLICY_COMBAT.md).
This script does not start a training run unless invoked as main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HUNG_OUTDIR_NAME = "combat_ppo_obs_v1_bh_v1"
HUNG_COMBAT_ZIP = (
    f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
)
DEFAULT_OUTPUT_DIR = "output/combat_runenv_onpolicy"
PROTOCOL_ID = "hang_protocol_runenv_onpolicy_combat LOCKED 2026-09-22"


def refuse_overwrite_hung_zip(output_dir: str | Path) -> Path:
    """Never write into the hung ``bh_v1`` outdir."""
    out = Path(output_dir).expanduser()
    parts = set(out.parts)
    if out.name == HUNG_OUTDIR_NAME or HUNG_OUTDIR_NAME in parts:
        raise SystemExit(
            f"refusing to overwrite hung combat zip outdir {out}; "
            f"pick a new --output-dir (e.g. {DEFAULT_OUTPUT_DIR})"
        )
    hung_parent = Path(HUNG_COMBAT_ZIP).expanduser().parent
    try:
        if out.resolve() == hung_parent.resolve():
            raise SystemExit(
                f"refusing to overwrite hung combat zip outdir {out}; "
                f"pick a new --output-dir (e.g. {DEFAULT_OUTPUT_DIR})"
            )
    except OSError:
        pass
    return out


def resolve_continue_from(path: str, *, must_exist: bool = True) -> Path:
    zip_path = Path(path).expanduser()
    if must_exist and not zip_path.is_file():
        raise SystemExit(f"continue-from zip not found: {path}")
    return zip_path


def model_obs_dim(model: Any) -> int:
    try:
        return int(model.observation_space.shape[0])
    except Exception as e:
        raise SystemExit(f"cannot read model observation_space: {e}") from e


def require_combat_obs_dim(model: Any, expected: int | None = None) -> int:
    """Refuse a zip that is not combat obs_v1 (bh_v1 / OBS_SIZE=181)."""
    from sts2_env.gym_env.observation import OBS_SIZE

    expected = OBS_SIZE if expected is None else int(expected)
    obs_dim = model_obs_dim(model)
    if obs_dim != expected:
        raise SystemExit(
            f"continue-from obs_dim={obs_dim} != combat OBS_SIZE={expected}; "
            "RunEnv zips and old 131-dim combat zips are not valid here"
        )
    return obs_dim


def require_train_deps():
    """Import SB3 stack. Cloud/CI without torch must not import this module's train()."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise SystemExit(
            "Training requires sb3-contrib and stable-baselines3. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from e
    return MaskablePPO, ActionMasker, DummyVecEnv, SubprocVecEnv, EvalCallback


def make_onpolicy_env(
    *,
    seed: int = 0,
    max_steps: int = 2000,
    jev_adapter: Any | None = None,
):
    """Factory for one hang-protocol combat-only env (no loadout fixtures)."""
    from sts2_env.gym_env.runenv_onpolicy_combat import RunEnvOnPolicyCombatEnv

    def _init():
        env = RunEnvOnPolicyCombatEnv(
            max_steps=max_steps,
            seed_offset=seed,
            jev_adapter=jev_adapter,
        )
        return env

    return _init


def make_masked_env(
    seed: int,
    *,
    max_steps: int = 2000,
    ActionMasker: Any = None,
):
    def mask_fn(env):
        return env.action_masks()

    def _init():
        env = make_onpolicy_env(seed=seed, max_steps=max_steps)()
        if ActionMasker is not None:
            env = ActionMasker(env, mask_fn)
        return env

    return _init


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune hung combat MaskablePPO on hang-protocol STS2RunEnv "
            "combat segments only (no loadout fixtures)"
        )
    )
    parser.add_argument(
        "--continue-from",
        "--model",
        dest="continue_from",
        default=HUNG_COMBAT_ZIP,
        help=(
            "Existing combat MaskablePPO zip (obs_v1=181). "
            f"Default: {HUNG_COMBAT_ZIP}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"New outdir (must not be {HUNG_OUTDIR_NAME}; default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2048,
        help="Combat PPO timesteps (default 2048 CPU smoke; Surplus full e.g. 500000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Parallel envs (default 1 for box CPU)",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Fine-tune LR (default 3e-5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size (default 64)")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="PPO rollout steps per env (combat steps; default 128)",
    )
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs (default 4)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Eval every N combat steps (0=skip; default 0 for smoke)",
    )
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000, help="Inner RunEnv step cap")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct hang env, refuse bh_v1 outdir, print protocol; do not load SB3/learn",
    )
    args = parser.parse_args(argv)
    if hasattr(args, "loadout"):
        raise SystemExit("loadout flags are forbidden on this trainer")
    return args


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """Prove hang wiring + combat-only env without torch / learn()."""
    from sts2_env.core.constants import ACTION_SPACE_SIZE
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.runenv_onpolicy_combat import (
        HANG_JEV,
        HANG_JEV_EVENT,
        HANG_JEV_NEOW,
        HANG_START_WITH_NEOW,
        hang_jev_flags,
    )

    out = refuse_overwrite_hung_zip(args.output_dir)
    flags = hang_jev_flags()
    env = make_onpolicy_env(seed=0, max_steps=min(int(args.max_steps), 80))()
    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    report = {
        "protocol": PROTOCOL_ID,
        "dry_run": True,
        "continue_from": args.continue_from,
        "output_dir": str(out),
        "n_envs": args.n_envs,
        "total_timesteps": args.total_timesteps,
        "hang": {
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "start_with_neow": HANG_START_WITH_NEOW,
            "allows_event": flags.allows_event(),
            "allows_neow": flags.allows_neow(),
        },
        "obs_shape": list(obs.shape),
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SPACE_SIZE,
        "mask_size": int(mask.shape[-1]),
        "phase": info.get("phase"),
        "noncombat_auto_steps": info.get("noncombat_auto_steps"),
        "loadout": info.get("loadout"),
        "loadout_forbidden": True,
    }
    env.close()
    return report


def train(args: argparse.Namespace) -> None:
    MaskablePPO, ActionMasker, DummyVecEnv, SubprocVecEnv, EvalCallback = (
        require_train_deps()
    )
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.runenv_onpolicy_combat import hang_jev_flags

    output_dir = refuse_overwrite_hung_zip(args.output_dir)
    continue_from = resolve_continue_from(args.continue_from, must_exist=True)
    flags = hang_jev_flags()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training MaskablePPO on hang-protocol RunEnv combat segments")
    print("  continue_from:   ", continue_from)
    print("  output_dir:      ", output_dir)
    print("  n_envs:          ", args.n_envs)
    print("  total_timesteps: ", args.total_timesteps)
    print("  n_steps:         ", args.n_steps)
    print("  learning_rate:   ", args.lr)
    print("  hang jev:        ", "on", "event", "off", "neow", "off", "start_with_neow")
    print("  allows_event:    ", flags.allows_event())
    print("  allows_neow:     ", flags.allows_neow())
    print("  loadout:         forbidden (on-policy RunEnv; not mix_neow_v1)")
    print()

    n_envs = max(1, int(args.n_envs))
    makers = [
        make_masked_env(i, max_steps=args.max_steps, ActionMasker=ActionMasker)
        for i in range(n_envs)
    ]
    if n_envs > 1:
        train_env = SubprocVecEnv(makers)
    else:
        train_env = DummyVecEnv(makers)

    model = MaskablePPO.load(str(continue_from), env=train_env, device="cpu")
    require_combat_obs_dim(model, OBS_SIZE)

    callback = None
    eval_env = None
    if int(args.eval_freq) > 0:
        eval_env = DummyVecEnv(
            [make_masked_env(9999, max_steps=args.max_steps, ActionMasker=ActionMasker)]
        )
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=max(args.eval_freq // n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=False,
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
    if eval_env is not None:
        eval_env.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        report = dry_run(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
