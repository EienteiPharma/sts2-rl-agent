"""Train a MaskablePPO agent on STS2 combat.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py
    python scripts/train_combat.py --loadout neow_early
    python scripts/train_combat.py --loadout mix_neow_v1

Requires: stable-baselines3, sb3-contrib, torch

Loadouts (``LOADOUT_SUITES``):

* ``bare`` — Ironclad starter
* ``neow_early`` — glob ``scripts/fixtures/neow_early/neow_early_*.json``
* ``loadout_v1`` — glob ``scripts/fixtures/loadout_v1/loadout_v1_*.json``
* ``mix_neow_v1`` — 50% neow_early + 50% loadout_v1 interleaved each reset

Aliases of mix_neow_v1: ``mix`` | ``mix_neow`` | ``mid_early`` | ``early_mid``.

This script does not start a training run unless invoked as main.
Do not change the hung combat eval zip from here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
NEOW_EARLY_FIXTURE_DIR = SCRIPTS_DIR / "fixtures" / "neow_early"
LOADOUT_V1_FIXTURE_DIR = SCRIPTS_DIR / "fixtures" / "loadout_v1"
NEOW_EARLY_LABEL_NEEDLE = "Neow+early"
SYNTH_ENTRY_HP_RATIO = 0.55

LOADOUT_SUITES: dict[str, dict[str, Any]] = {
    "bare": {},
    "neow_early": {"glob": "fixtures/neow_early/neow_early_*.json"},
    "loadout_v1": {"glob": "fixtures/loadout_v1/loadout_v1_*.json"},
    "mix_neow_v1": {"mix": ("neow_early", "loadout_v1")},
}

LOADOUT_ALIASES: dict[str, str] = {
    "mix": "mix_neow_v1",
    "mix_neow": "mix_neow_v1",
    "mid_early": "mix_neow_v1",
    "early_mid": "mix_neow_v1",
}


def resolve_loadout(name: str) -> str:
    resolved = LOADOUT_ALIASES.get(name, name)
    if resolved not in LOADOUT_SUITES:
        raise SystemExit(f"unknown loadout: {name}")
    return resolved


def synth_entry_hp(snapshot_hp: int, max_hp: int) -> tuple[int, bool]:
    """Combat entry HP from a death snapshot. hp==0 → ~0.55 * max_hp."""
    max_hp = max(int(max_hp), 1)
    if int(snapshot_hp) <= 0:
        return max(1, int(round(SYNTH_ENTRY_HP_RATIO * max_hp))), True
    return int(snapshot_hp), False


def fixture_glob_paths(suite: str, *, neow_early_dir: Path | None = None) -> list[Path]:
    spec = LOADOUT_SUITES[suite]
    pattern = spec.get("glob")
    if not pattern:
        return []
    if suite == "neow_early":
        root = Path(neow_early_dir) if neow_early_dir is not None else NEOW_EARLY_FIXTURE_DIR
        return sorted(root.glob("neow_early_*.json"))
    return sorted((SCRIPTS_DIR / pattern).parent.glob(Path(pattern).name))


def load_json_fixtures(paths: list[Path]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"fixture is not an object: {path}")
        fixtures.append(data)
    return fixtures


def load_neow_early_fixtures(fixture_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(fixture_dir) if fixture_dir is not None else NEOW_EARLY_FIXTURE_DIR
    paths = sorted(root.glob("neow_early_*.json"))
    fixtures = load_json_fixtures(paths)
    if not fixtures:
        raise SystemExit(f"no neow_early fixtures matching {root / 'neow_early_*.json'}")
    return fixtures


def load_loadout_v1_fixtures() -> list[dict[str, Any]]:
    paths = fixture_glob_paths("loadout_v1")
    fixtures = load_json_fixtures(paths)
    if not fixtures:
        raise SystemExit(f"no loadout_v1 fixtures matching {LOADOUT_V1_FIXTURE_DIR / 'loadout_v1_*.json'}")
    return fixtures


def load_suite_fixtures(suite: str, *, neow_early_dir: Path | None = None) -> list[dict[str, Any]]:
    if suite == "neow_early":
        fixtures = load_neow_early_fixtures(neow_early_dir)
        assert_neow_early_labels(fixtures)
        return fixtures
    if suite == "loadout_v1":
        return load_loadout_v1_fixtures()
    raise SystemExit(f"suite {suite!r} has no fixture glob")


def assert_neow_early_labels(fixtures: list[dict[str, Any]]) -> None:
    for i, fx in enumerate(fixtures):
        label = str(fx.get("label") or "")
        if NEOW_EARLY_LABEL_NEEDLE not in label:
            raise SystemExit(
                f"neow_early fixture {i} label must say Neow+early, not bare: {label!r}"
            )


def materialize_fixture(fixture: dict[str, Any], *, suite: str | None = None) -> dict[str, Any]:
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
        if entry is None:
            raise SystemExit("fixture deck contains null entry")
        if isinstance(entry, dict):
            # loadout_v1 JSON uses {"id": CARD, "upgraded": bool}; also accept card_id
            raw = entry.get("card_id", entry.get("id"))
            upgraded = bool(entry.get("upgraded"))
        else:
            raw = entry
            upgraded = False
        if raw is None:
            raise SystemExit(f"unknown card_id in fixture: {raw!r}")
        name = str(raw)
        if name not in CardId.__members__:
            raise SystemExit(f"unknown card_id in fixture: {name}")
        deck.append(create_card(CardId[name], upgraded=upgraded))
    if not deck:
        raise SystemExit("fixture has empty deck")
    return {
        "deck": deck,
        "hp": hp,
        "max_hp": max_hp,
        "label": fixture.get("label"),
        "suite": suite or fixture.get("suite"),
    }


def materialize_neow_early_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    return materialize_fixture(fixture, suite="neow_early")


class RotatingNeowEarlyProvider:
    """Rotate LOCKED Neow+early fixtures on each combat reset."""

    def __init__(self, fixtures: list[dict[str, Any]], offset: int = 0):
        self.fixtures = fixtures
        self.offset = int(offset)
        self.i = 0

    def __call__(self) -> dict[str, Any]:
        fx = self.fixtures[(self.offset + self.i) % len(self.fixtures)]
        self.i += 1
        return materialize_fixture(fx, suite="neow_early")


class RotatingSuiteProvider:
    """Rotate one fixture suite on each combat reset."""

    def __init__(self, fixtures: list[dict[str, Any]], suite: str, offset: int = 0):
        self.fixtures = fixtures
        self.suite = suite
        self.offset = int(offset)
        self.i = 0

    def __call__(self) -> dict[str, Any]:
        fx = self.fixtures[(self.offset + self.i) % len(self.fixtures)]
        self.i += 1
        return materialize_fixture(fx, suite=self.suite)


class InterleavedMixProvider:
    """50/50 interleave of named suites each reset (even/odd)."""

    def __init__(
        self,
        named_suites: list[tuple[str, list[dict[str, Any]]]],
        offset: int = 0,
    ):
        if len(named_suites) < 2:
            raise SystemExit("mix loadout needs at least two suites")
        self.named_suites = named_suites
        self.offset = int(offset)
        self.i = 0

    def __call__(self) -> dict[str, Any]:
        n = len(self.named_suites)
        suite_i = (self.offset + self.i) % n
        name, fixtures = self.named_suites[suite_i]
        fx = fixtures[((self.offset + self.i) // n) % len(fixtures)]
        self.i += 1
        return materialize_fixture(fx, suite=name)


def make_loadout_provider(
    loadout: str,
    *,
    offset: int = 0,
    fixture_dir: Path | None = None,
) -> Callable[[], dict[str, Any]] | None:
    loadout = resolve_loadout(loadout)
    spec = LOADOUT_SUITES[loadout]
    if loadout == "bare" or not spec:
        return None
    mix = spec.get("mix")
    if mix:
        named = []
        for name in mix:
            named.append((name, load_suite_fixtures(name, neow_early_dir=fixture_dir)))
        return InterleavedMixProvider(named, offset=offset)
    if loadout == "neow_early":
        fixtures = load_suite_fixtures("neow_early", neow_early_dir=fixture_dir)
        return RotatingNeowEarlyProvider(fixtures, offset=offset)
    fixtures = load_suite_fixtures(loadout, neow_early_dir=fixture_dir)
    return RotatingSuiteProvider(fixtures, suite=loadout, offset=offset)


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

    loadout = resolve_loadout(getattr(args, "loadout", "bare"))
    args.loadout = loadout
    fixture_dir = Path(getattr(args, "loadout_dir", NEOW_EARLY_FIXTURE_DIR))
    n_fixtures = 0
    mix_note = ""
    spec = LOADOUT_SUITES[loadout]
    if loadout == "neow_early":
        fixtures = load_neow_early_fixtures(fixture_dir)
        assert_neow_early_labels(fixtures)
        n_fixtures = len(fixtures)
    elif loadout == "loadout_v1":
        n_fixtures = len(load_loadout_v1_fixtures())
    elif spec.get("mix"):
        parts = []
        for name in spec["mix"]:
            n = len(load_suite_fixtures(name, neow_early_dir=fixture_dir))
            parts.append(f"{name}={n}")
        mix_note = "50/50 interleave " + " + ".join(parts)

    print(f"Training MaskablePPO on STS2 combat")
    if loadout == "neow_early":
        print(
            f"  loadout:         neow_early "
            f"(Neow+early natural Act1 decks; NOT bare starter; {n_fixtures} fixtures)"
        )
    elif loadout == "loadout_v1":
        print(f"  loadout:         loadout_v1 (mid-act combat suite; {n_fixtures} fixtures)")
    elif spec.get("mix"):
        print(
            f"  loadout:         mix_neow_v1 "
            f"(50% neow_early + 50% loadout_v1 interleaved; {mix_note})"
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
                loadout,
                offset=seed,
                fixture_dir=fixture_dir if loadout in {"neow_early", "mix_neow_v1"} else None,
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
        choices=sorted(set(LOADOUT_SUITES) | set(LOADOUT_ALIASES)),
        default="bare",
        help=(
            "Combat start deck. bare=Ironclad starter. "
            "neow_early=Neow+early natural Act1 death snapshots (NOT bare). "
            "loadout_v1=mid-act combat suite. "
            "mix_neow_v1=50%% neow_early + 50%% loadout_v1 interleaved "
            "(aliases: mix, mix_neow, mid_early, early_mid)."
        ),
    )
    parser.add_argument(
        "--loadout-dir",
        type=str,
        default=str(NEOW_EARLY_FIXTURE_DIR),
        help="Directory of neow_early LOCKED JSON fixtures",
    )
    args = parser.parse_args(argv)
    args.loadout = resolve_loadout(args.loadout)
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
