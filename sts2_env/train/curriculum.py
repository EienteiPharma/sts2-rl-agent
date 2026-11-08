"""Starting conditions for one training run.

A run either begins from random weights or continues an earlier run-env
checkpoint. The start mode is a single value, so the trainer never holds a
path string beside a flag that has to agree with it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium

NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# SB3 rejects a mismatched env inside ``load``, before a model exists for us to
# compare. These are the two messages it raises from ``check_for_correct_spaces``.
_SB3_SPACE_ERRORS = (
    "Observation spaces do not match",
    "Action spaces do not match",
)


class CheckpointSpaceMismatch(ValueError):
    """The checkpoint's networks cannot read this environment's spaces."""

    def __init__(self, path: Path, detail: str = "") -> None:
        message = f"{path} was not trained on this environment's spaces"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.path = path


class HasSpaces(Protocol):
    observation_space: gymnasium.Space
    action_space: gymnasium.Space


@dataclass(frozen=True)
class FreshStart:
    """Train a new model from random weights."""

    def banner_path(self) -> str:
        return "(fresh)"


@dataclass(frozen=True)
class CurriculumLoad:
    """Fine-tune a checkpoint saved by an earlier curriculum phase."""

    path: Path

    def banner_path(self) -> str:
        return str(self.path)


TrainStart = FreshStart | CurriculumLoad


@dataclass(frozen=True)
class TrainHyperparams:
    """PPO settings the command line owns for this run."""

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float

    def ppo_kwargs(self) -> dict[str, Any]:
        """Settings that override whatever a loaded checkpoint saved."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
        }


def parse_checkpoint(raw: str) -> CurriculumLoad:
    """Turn a command-line path into a load instruction, or say why it cannot."""
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"checkpoint is not readable: {path}")
    return CurriculumLoad(path)


def assert_spaces_match(path: Path, checkpoint: HasSpaces, env: HasSpaces) -> None:
    """Reject a checkpoint whose networks were sized for another env."""
    if checkpoint.observation_space != env.observation_space:
        raise CheckpointSpaceMismatch(
            path,
            f"observation space {checkpoint.observation_space} "
            f"!= {env.observation_space}",
        )
    if checkpoint.action_space != env.action_space:
        raise CheckpointSpaceMismatch(
            path,
            f"action space {checkpoint.action_space} != {env.action_space}",
        )


def load_maskable_checkpoint(path: Path, env: Any, **load_kwargs: Any) -> Any:
    """Load a run-env zip onto ``env``, or fail closed on a space mismatch.

    Passing ``env`` makes SB3 check both spaces and set ``n_envs`` to this
    env's count, which is 1 for a bare run env. A zip from a checkout with a
    different observation width raises here instead of producing numbers.

    A run env reads the pin first. A zip with no pin, a reordered
    catalog, or another embed dim raises
    :class:`~sts2_env.train.card_embed.CheckpointVocabMismatch` before SB3
    builds a network. There is no pad path and no flat-extractor fallback.

    Loading is pinned to CPU because eval and inference run on CPU; a trainer
    moves the weights to the update device itself.
    """
    from sb3_contrib import MaskablePPO

    from sts2_env.train.card_embed import (
        RunEmbedPin,
        assert_card_vocab_match,
        assert_teacher_vocab_match,
        is_run_observation_space,
        is_teacher_observation_space,
        stored_observation_space,
    )

    if is_run_observation_space(env.observation_space) or is_teacher_observation_space(
        env.observation_space
    ):
        stored_space = stored_observation_space(Path(path))
        if stored_space is not None and stored_space != env.observation_space:
            raise CheckpointSpaceMismatch(
                path, f"observation space {stored_space} != {env.observation_space}"
            )
        if is_teacher_observation_space(env.observation_space):
            assert_teacher_vocab_match(Path(path), RunEmbedPin.live())
        else:
            assert_card_vocab_match(Path(path), RunEmbedPin.live())

    try:
        model = MaskablePPO.load(str(path), env=env, device="cpu", **load_kwargs)
    except ValueError as exc:
        if not any(marker in str(exc) for marker in _SB3_SPACE_ERRORS):
            raise
        raise CheckpointSpaceMismatch(path, str(exc)) from exc
    assert_spaces_match(path, model, env)
    return model
