#!/usr/bin/env python3
"""Anti-forgetting mix: hang-protocol RunEnv combat + loadout_v1 fixtures.

Pure RunEnv on-policy 500k (``combat_runenv_onpolicy_v1``) is FROZEN:
RunEnv 4%/med7 but loadout_v1 HOLD FAIL. This trainer continues ``bh_v1``
and interleaves hang RunEnv combat segments with loadout_v1 so HOLD
does not collapse.

Usage:
    python scripts/train_combat_runenv_antiforget.py --dry-run
    python scripts/train_combat_runenv_antiforget.py \\
        --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \\
        --output-dir output/combat_runenv_antiforget_smoke \\
        --runenv-frac 0.7 --n-envs 1 --n-steps 64 --total-timesteps 256

Never overwrites ``bh_v1`` or ``combat_runenv_onpolicy_v1``.
Hang eval flags/bars unchanged. Not a win claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sts2_env.eval.combat_hold import (
    HOLD_BOSS_MIN,
    HOLD_OVERALL_MIN,
    run_hold_smoke,
)
from sts2_env.gym_env.runenv_antiforget import (
    DEFAULT_RUNENV_FRAC,
    MixedHangLoadoutEnv,
    make_loadout_v1_provider,
    parse_runenv_frac,
)
from sts2_env.gym_env.runenv_onpolicy_combat import (
    HANG_JEV,
    HANG_JEV_EVENT,
    HANG_JEV_NEOW,
    HANG_START_WITH_NEOW,
    hang_jev_flags,
)

HUNG_OUTDIR_NAME = "combat_ppo_obs_v1_bh_v1"
ONPOLICY_FROZEN_OUTDIR = "combat_runenv_onpolicy_v1"
FROZEN_OUTDIR_NAMES = (HUNG_OUTDIR_NAME, ONPOLICY_FROZEN_OUTDIR)
HUNG_COMBAT_ZIP = f"/workspace/sts2-sim/output/{HUNG_OUTDIR_NAME}/final_model.zip"
DEFAULT_OUTPUT_DIR = "output/combat_runenv_antiforget"
PROTOCOL_ID = "hang_protocol_runenv_antiforget LOCKED 2026-09-22"


def refuse_overwrite_frozen(output_dir: str | Path) -> Path:
    out = Path(output_dir).expanduser()
    parts = set(out.parts)
    for name in FROZEN_OUTDIR_NAMES:
        if out.name == name or name in parts:
            raise SystemExit(
                f"refusing to overwrite frozen outdir {out}; "
                f"pick a new --output-dir (e.g. {DEFAULT_OUTPUT_DIR})"
            )
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


def make_mixed_env(
    *,
    seed: int = 0,
    runenv_frac: float = DEFAULT_RUNENV_FRAC,
    max_steps: int = 2000,
    jev_adapter: Any | None = None,
):
    def _init():
        return MixedHangLoadoutEnv(
            runenv_frac=runenv_frac,
            loadout_provider=make_loadout_v1_provider(offset=seed),
            max_steps=max_steps,
            seed_offset=seed,
            jev_adapter=jev_adapter,
        )

    return _init


def make_masked_env(
    seed: int,
    *,
    runenv_frac: float,
    max_steps: int = 2000,
    ActionMasker: Any = None,
):
    def mask_fn(env):
        return env.action_masks()

    def _init():
        env = make_mixed_env(
            seed=seed, runenv_frac=runenv_frac, max_steps=max_steps
        )()
        if ActionMasker is not None:
            env = ActionMasker(env, mask_fn)
        return env

    return _init


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune bh_v1 on hang RunEnv combat mixed with loadout_v1 "
            "(anti-forgetting). Pure onpolicy_v1 500k is frozen."
        )
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
        help=f"New outdir (not {HUNG_OUTDIR_NAME} / {ONPOLICY_FROZEN_OUTDIR})",
    )
    parser.add_argument(
        "--runenv-frac",
        type=float,
        default=DEFAULT_RUNENV_FRAC,
        help="P(hang RunEnv episode) on reset (default 0.7; rest is loadout_v1)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2048,
        help="Combat PPO timesteps (default 2048 CPU smoke; first full recipe 250000)",
    )
    parser.add_argument("--n-envs", type=int, default=1)
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
    parser.add_argument(
        "--hold-n-eps",
        type=int,
        default=1,
        help="HOLD episodes per fixture×encounter (1=smoke, 20=full box HOLD)",
    )
    parser.add_argument(
        "--hold-stop",
        action="store_true",
        help="Stop learn() if a HOLD smoke misses overall≥70 / Boss≥40",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct mix env + HOLD jobs; no SB3 learn",
    )
    args = parser.parse_args(argv)
    args.runenv_frac = parse_runenv_frac(args.runenv_frac)
    return args


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    from sts2_env.core.constants import ACTION_SPACE_SIZE
    from sts2_env.eval.combat_hold import hold_jobs
    from sts2_env.gym_env.observation import OBS_SIZE

    out = refuse_overwrite_frozen(args.output_dir)
    flags = hang_jev_flags()
    env = make_mixed_env(
        seed=0,
        runenv_frac=1.0,
        max_steps=min(int(args.max_steps), 80),
    )()
    obs_r, info_r = env.reset(seed=0)
    env.close()
    env_l = make_mixed_env(
        seed=1,
        runenv_frac=0.0,
        max_steps=40,
    )()
    obs_l, info_l = env_l.reset(seed=1)
    mask_l = env_l.action_masks()
    env_l.close()
    jobs = hold_jobs(n_eps=1)
    return {
        "protocol": PROTOCOL_ID,
        "dry_run": True,
        "continue_from": args.continue_from,
        "output_dir": str(out),
        "runenv_frac": args.runenv_frac,
        "loadout_frac": round(1.0 - args.runenv_frac, 4),
        "loadout_half": "loadout_v1",
        "mix_neow_v1": False,
        "n_envs": args.n_envs,
        "total_timesteps": args.total_timesteps,
        "lr": args.lr,
        "hang": {
            "jev": HANG_JEV,
            "jev_event": HANG_JEV_EVENT,
            "jev_neow": HANG_JEV_NEOW,
            "start_with_neow": HANG_START_WITH_NEOW,
            "allows_event": flags.allows_event(),
            "allows_neow": flags.allows_neow(),
        },
        "runenv_reset": {
            "mix_source": info_r.get("mix_source"),
            "phase": info_r.get("phase"),
            "obs_size": int(np_shape0(obs_r)),
            "start_with_neow": info_r.get("start_with_neow"),
            "jev_event": info_r.get("jev_event"),
            "jev_neow": info_r.get("jev_neow"),
        },
        "loadout_reset": {
            "mix_source": info_l.get("mix_source"),
            "obs_size": int(np_shape0(obs_l)),
            "mask_size": int(mask_l.shape[-1]),
            "loadout": info_l.get("loadout"),
        },
        "hold": {
            "overall_min": HOLD_OVERALL_MIN,
            "boss_min": HOLD_BOSS_MIN,
            "jobs_n_eps1": len(jobs),
            "hold_freq": args.hold_freq,
            "hold_n_eps": args.hold_n_eps,
            "hold_stop": bool(args.hold_stop),
        },
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SPACE_SIZE,
        "frozen_outdirs": list(FROZEN_OUTDIR_NAMES),
        "pure_onpolicy_500k": "frozen",
    }


def np_shape0(obs) -> int:
    import numpy as np

    return int(np.asarray(obs).shape[-1])


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
            log_path = output_dir / "hold_logs.json"
            log_path.write_text(json.dumps(self.history, indent=2) + "\n")
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
                print(f"  saved best-by-HOLD → {dest}")
            if stop_on_fail and not passed:
                print("HOLD smoke missed gate; --hold-stop ending learn()")
                return False
            return True

    return HoldCheckpointCallback()


def train(args: argparse.Namespace) -> None:
    MaskablePPO, ActionMasker, DummyVecEnv, SubprocVecEnv, BaseCallback = (
        require_train_deps()
    )
    from sts2_env.gym_env.observation import OBS_SIZE

    output_dir = refuse_overwrite_frozen(args.output_dir)
    continue_from = resolve_continue_from(args.continue_from, must_exist=True)
    flags = hang_jev_flags()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training MaskablePPO anti-forgetting mix")
    print("  continue_from:   ", continue_from)
    print("  output_dir:      ", output_dir)
    print("  runenv_frac:     ", args.runenv_frac, "(loadout_v1", round(1.0 - args.runenv_frac, 4), ")")
    print("  n_envs:          ", args.n_envs)
    print("  total_timesteps: ", args.total_timesteps)
    print("  learning_rate:   ", args.lr)
    print("  hang:            jev on / event off / neow off / start_with_neow")
    print("  allows_event:    ", flags.allows_event())
    print("  allows_neow:     ", flags.allows_neow())
    print("  hold_freq:       ", args.hold_freq, "n_eps", args.hold_n_eps)
    print("  frozen:          ", FROZEN_OUTDIR_NAMES)
    print()

    n_envs = max(1, int(args.n_envs))
    makers = [
        make_masked_env(
            i,
            runenv_frac=args.runenv_frac,
            max_steps=args.max_steps,
            ActionMasker=ActionMasker,
        )
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(makers) if n_envs > 1 else DummyVecEnv(makers)

    model = MaskablePPO.load(str(continue_from), env=train_env, device="cpu")
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
        report = dry_run(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
