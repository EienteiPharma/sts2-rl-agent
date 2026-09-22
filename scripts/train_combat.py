"""Train a MaskablePPO agent on STS2 combat.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py
    python scripts/train_combat.py --loadout neow_early

Requires: stable-baselines3, sb3-contrib, torch

``--loadout neow_early`` rotates LOCKED JSON fixtures under
``scripts/fixtures/neow_early/`` sampled from RunEnv death snapshots
(Neow+early natural Act1 decks, NOT bare starter). This script does not
start a training run unless invoked as main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

NEOW_EARLY_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "neow_early"
NEOW_EARLY_LABEL_NEEDLE = "Neow+early"
SYNTH_ENTRY_HP_RATIO = 0.55


def synth_entry_hp(snapshot_hp: int, max_hp: int) -> tuple[int, bool]:
    """Combat entry HP from a death snapshot. hp==0 → ~0.55 * max_hp."""
    max_hp = max(int(max_hp), 1)
    if int(snapshot_hp) <= 0:
        return max(1, int(round(SYNTH_ENTRY_HP_RATIO * max_hp))), True
    return int(snapshot_hp), False


def load_neow_early_fixtures(fixture_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(fixture_dir) if fixture_dir is not None else NEOW_EARLY_FIXTURE_DIR
    paths = sorted(root.glob("*.json"))
    fixtures: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"neow_early fixture is not an object: {path}")
        fixtures.append(data)
    if not fixtures:
        raise SystemExit(f"no neow_early fixtures under {root}")
    return fixtures


def assert_neow_early_labels(fixtures: list[dict[str, Any]]) -> None:
    for i, fx in enumerate(fixtures):
        label = str(fx.get("label") or "")
        if NEOW_EARLY_LABEL_NEEDLE not in label:
            raise SystemExit(
                f"neow_early fixture {i} label must say Neow+early, not bare: {label!r}"
            )
        if "bare" in label.lower() and NEOW_EARLY_LABEL_NEEDLE not in label:
            raise SystemExit(f"neow_early fixture {i} labelled bare: {label!r}")


def materialize_neow_early_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    from sts2_env.cards.factory import create_card
    from sts2_env.core.enums import CardId

    max_hp = int(fixture.get("max_hp") or 80)
    snapshot_hp = int(fixture.get("snapshot_hp", fixture.get("hp", 0)))
    stored_hp = int(fixture.get("hp", snapshot_hp))
    if stored_hp <= 0:
        hp, _ = synth_entry_hp(snapshot_hp, max_hp)
    else:
        hp = stored_hp
    deck = []
    for entry in fixture.get("deck") or []:
        raw = entry.get("card_id") if isinstance(entry, dict) else entry
        upgraded = bool(entry.get("upgraded")) if isinstance(entry, dict) else False
        name = str(raw)
        if name not in CardId.__members__:
            raise SystemExit(f"unknown card_id in neow_early fixture: {name}")
        deck.append(create_card(CardId[name], upgraded=upgraded))
    if not deck:
        raise SystemExit("neow_early fixture has empty deck")
    return {
        "deck": deck,
        "hp": hp,
        "max_hp": max_hp,
        "label": fixture.get("label"),
    }


class RotatingNeowEarlyProvider:
    """Rotate LOCKED Neow+early fixtures on each combat reset."""

    def __init__(self, fixtures: list[dict[str, Any]], offset: int = 0):
        self.fixtures = fixtures
        self.offset = int(offset)
        self.i = 0

    def __call__(self) -> dict[str, Any]:
        fx = self.fixtures[(self.offset + self.i) % len(self.fixtures)]
        self.i += 1
        return materialize_neow_early_fixture(fx)


def make_loadout_provider(
    loadout: str,
    *,
    offset: int = 0,
    fixture_dir: Path | None = None,
) -> Callable[[], dict[str, Any]] | None:
    if loadout == "bare":
        return None
    if loadout == "neow_early":
        fixtures = load_neow_early_fixtures(fixture_dir)
        assert_neow_early_labels(fixtures)
        return RotatingNeowEarlyProvider(fixtures, offset=offset)
    raise SystemExit(f"unknown loadout: {loadout}")


def make_env(seed: int = 0, loadout: str = "bare"):
    """Create a single STS2CombatEnv."""
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    def _init():
        env = STS2CombatEnv(loadout_provider=make_loadout_provider(loadout, offset=seed))
        env.reset(seed=seed)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.gym_env.combat_env import STS2CombatEnv

    loadout = getattr(args, "loadout", "bare")
    fixture_dir = Path(getattr(args, "loadout_dir", NEOW_EARLY_FIXTURE_DIR))
    n_fixtures = 0
    if loadout == "neow_early":
        fixtures = load_neow_early_fixtures(fixture_dir)
        assert_neow_early_labels(fixtures)
        n_fixtures = len(fixtures)

    print(f"Training MaskablePPO on STS2 combat")
    if loadout == "neow_early":
        print(
            f"  loadout:         neow_early "
            f"(Neow+early natural Act1 decks; NOT bare starter; {n_fixtures} fixtures)"
        )
    else:
        print(f"  loadout:         bare (Ironclad starter deck)")
    print(f"  n_envs:          {args.n_envs}")
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  learning_rate:   {args.lr}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  output_dir:      {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap env with action masker
    def mask_fn(env):
        return env.action_masks()

    def make_masked_env(seed: int):
        def _init():
            provider = make_loadout_provider(
                loadout, offset=seed, fixture_dir=fixture_dir if loadout == "neow_early" else None
            )
            env = STS2CombatEnv(loadout_provider=provider)
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    # Create vectorized envs
    if args.n_envs > 1:
        train_env = SubprocVecEnv([make_masked_env(i) for i in range(args.n_envs)])
    else:
        train_env = DummyVecEnv([make_masked_env(0)])

    # Eval env (always single) — same loadout, different rotation offset
    eval_env = DummyVecEnv([make_masked_env(9999)])

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
    )

    # Train
    start = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    # Save final model
    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    # Quick evaluation
    print("\n--- Final Evaluation ---")
    evaluate(model, n_episodes=100, loadout=loadout, fixture_dir=fixture_dir)

    train_env.close()
    eval_env.close()


def evaluate(model, n_episodes: int = 100, loadout: str = "bare", fixture_dir: Path | None = None):
    """Evaluate trained model."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    def mask_fn(env):
        return env.action_masks()

    provider = make_loadout_provider(loadout, offset=10_000, fixture_dir=fixture_dir)
    env = ActionMasker(STS2CombatEnv(loadout_provider=provider), mask_fn)
    wins = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)

    print(f"Episodes:    {n_episodes}")
    print(f"Win rate:    {wins / n_episodes:.1%}")
    print(f"Avg reward:  {np.mean(total_rewards):.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on STS2 combat")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size (default: 256)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout per env (default: 2048)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N steps (default: 10000)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Episodes per evaluation (default: 20)")
    parser.add_argument("--output-dir", type=str, default="output/combat_ppo",
                        help="Output directory (default: output/combat_ppo)")
    parser.add_argument(
        "--loadout",
        choices=["bare", "neow_early"],
        default="bare",
        help=(
            "Combat start deck. bare=Ironclad starter. "
            "neow_early=rotating Neow+early natural Act1 death snapshots "
            "(NOT bare)."
        ),
    )
    parser.add_argument(
        "--loadout-dir",
        type=str,
        default=str(NEOW_EARLY_FIXTURE_DIR),
        help="Directory of neow_early LOCKED JSON fixtures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
