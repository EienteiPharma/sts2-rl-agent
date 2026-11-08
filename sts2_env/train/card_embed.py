"""Learned card, relic, potion, boss, and monster tables for the full-run policy.

The run env still writes ``Box(RUN_OBS_SIZE)`` floats and ``Discrete(157)``
still names slots. What changes is what the policy reads: every catalog float
in that vector becomes an ``nn.Embedding`` row, so two relics or two monsters
are two rows instead of two nearby catalog floats.

Each card bag segment (deck, draw, discard, exhaust) becomes one token per
slot, the card row plus its upgrade bit, pooled by sum and by mean over the
occupied slots. The pool has no order, so the draw order stays hidden.

One rule keeps the float and the row in step. Each catalog writes
``(index + 1) / (size + 1)``; :class:`CardEmbedExtractor` reads
``round(norm * divisor) - 1`` and adds one for the pad row. Each divisor is a
constructor argument, so a checkpoint pin records the formula and not only the
names. A checkout that rewrites one catalog formula cannot load the other
table.

Row ``0`` is pad and stays zero. It is what an empty slot writes. A live id
never reaches it.

Only ``Box(RUN_OBS_SIZE)`` attaches this extractor. A combat ``Box(131)`` job
keeps its plain MLP.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE
from sts2_env.core.enums import CardId
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    CARD_IDS,
    CARD_ID_NORM_DIVISOR,
    HAND_ID_FIELD,
    HAND_START,
    NUM_CARD_IDS,
)
from sts2_env.gym_env.run_observation import (
    BOSS_ID_NORM_DIVISOR,
    BOSS_IDS,
    ENEMY_ID_FIELD,
    ENEMY_SLOT_FIELDS,
    LAYOUT,
    LAYOUT_V1,
    MONSTER_ID_NORM_DIVISOR,
    MONSTER_IDS,
    OPTION_BLOCK,
    PILE_SLOTS,
    PILES,
    POTION_ID_NORM_DIVISOR,
    POTION_IDS,
    RELIC_ID_NORM_DIVISOR,
    RELICS,
    RUN_OBS_SIZE,
    RUN_OBS_SIZE_V1,
    OptionKind,
)
from sts2_env.relics.base import RelicId

__all__ = [
    "CARD_EMBED_DIM",
    "CARD_KINDS",
    "POTION_KINDS",
    "RELIC_KINDS",
    "BossVocab",
    "CardEmbedExtractor",
    "CardVocab",
    "CardVocabMismatch",
    "CatalogVocab",
    "CheckpointVocabMismatch",
    "EmbedLayout",
    "MonsterVocab",
    "PotionVocab",
    "RelicVocab",
    "RunEmbedPin",
    "assert_card_vocab_match",
    "assert_teacher_vocab_match",
    "card_embed_policy_kwargs",
    "is_run_observation_space",
    "is_teacher_observation_space",
    "stored_observation_space",
]

CARD_EMBED_DIM = 16
PAD_ROW = 0
CARD_KINDS = frozenset({OptionKind.CARD, OptionKind.BUNDLE, OptionKind.CHOOSE})
RELIC_KINDS = frozenset({OptionKind.RELIC, OptionKind.TREASURE})
POTION_KINDS = frozenset({OptionKind.POTION})

_PIN_KEYS_TEACHER = (
    "card_id_names",
    "card_embed_dim",
    "card_id_norm_divisor",
    "relic_id_names",
    "relic_id_norm_divisor",
    "potion_id_names",
    "potion_id_norm_divisor",
)
_PIN_KEYS = _PIN_KEYS_TEACHER + (
    "boss_id_names",
    "boss_id_norm_divisor",
    "monster_id_names",
    "monster_id_norm_divisor",
)

_OPTION_KIND_FIELD = 1
_OPTION_ID_FIELD = 2


class CardVocabMismatch(ValueError):
    """Live ``CardId`` order, embed dim, or norm divisor disagrees with the pin."""


class CheckpointVocabMismatch(CardVocabMismatch):
    """A zip carries no pin, or its pin does not match this checkout."""

    def __init__(self, path: Path, detail: str = "") -> None:
        message = f"{path} carries no card table this checkout can rebuild"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class CatalogVocab:
    """One catalog the pin stores: names, embed dim, and the norm divisor."""

    names: tuple[str, ...]
    embed_dim: int
    norm_divisor: int
    expected_size: int

    @staticmethod
    def parse(
        names: Sequence[str],
        embed_dim: int,
        norm_divisor: int,
        expected_size: int,
    ) -> "CatalogVocab":
        """Build a catalog from stored values, rejecting anything unusable."""
        stored = tuple(names)
        if len(stored) != expected_size:
            raise CardVocabMismatch(
                f"pin names {len(stored)} entries, this checkout has {expected_size}"
            )
        if any(not isinstance(name, str) for name in stored):
            raise CardVocabMismatch("pin names must all be strings")
        if embed_dim < 1:
            raise CardVocabMismatch(f"pin embed dim must be >= 1, got {embed_dim}")
        if norm_divisor < 1:
            raise CardVocabMismatch(f"pin norm divisor must be >= 1, got {norm_divisor}")
        return CatalogVocab(
            names=stored,
            embed_dim=int(embed_dim),
            norm_divisor=int(norm_divisor),
            expected_size=int(expected_size),
        )

    def assert_match(self, other: "CatalogVocab") -> None:
        """Raise :class:`CardVocabMismatch` on any field inequality."""
        if self.names != other.names:
            first = next(
                (
                    index
                    for index, (mine, theirs) in enumerate(zip(self.names, other.names))
                    if mine != theirs
                ),
                min(len(self.names), len(other.names)),
            )
            raise CardVocabMismatch(
                f"catalog order differs at index {first}: pin has "
                f"{self.names[first : first + 1]}, this checkout has "
                f"{other.names[first : first + 1]}"
            )
        if self.embed_dim != other.embed_dim:
            raise CardVocabMismatch(
                f"embed dim {self.embed_dim} != {other.embed_dim}"
            )
        if self.norm_divisor != other.norm_divisor:
            raise CardVocabMismatch(
                f"norm divisor {self.norm_divisor} != {other.norm_divisor}"
            )


@dataclass(frozen=True)
class CardVocab(CatalogVocab):
    """The card catalog a checkpoint stores beside its weights.

    ``names[i]`` is ``CARD_IDS[i].name``, so embedding row ``i + 1`` is that
    card. Row ``0`` is pad. A checkout that reorders, inserts, deletes, or
    renames a ``CardId`` builds a different ``names`` tuple, and one that
    rewrites the norm formula builds a different ``norm_divisor``. Neither can
    load an older table.
    """

    @staticmethod
    def live() -> "CardVocab":
        return CardVocab(
            names=tuple(card_id.name for card_id in CARD_IDS),
            embed_dim=CARD_EMBED_DIM,
            norm_divisor=CARD_ID_NORM_DIVISOR,
            expected_size=NUM_CARD_IDS,
        )

    @staticmethod
    def parse(names: Sequence[str], embed_dim: int, norm_divisor: int) -> "CardVocab":
        """Build a card pin from stored values, rejecting anything unusable."""
        base = CatalogVocab.parse(names, embed_dim, norm_divisor, NUM_CARD_IDS)
        return CardVocab(
            base.names, base.embed_dim, base.norm_divisor, base.expected_size
        )

    def embed_row(self, card_id: CardId) -> int:
        """Row of a live card. Never :data:`PAD_ROW`."""
        return CARD_IDS.index(card_id) + 1


@dataclass(frozen=True)
class RelicVocab(CatalogVocab):
    """The relic catalog a checkpoint stores beside its weights."""

    @staticmethod
    def live() -> "RelicVocab":
        return RelicVocab(
            names=tuple(relic.name for relic in RELICS),
            embed_dim=CARD_EMBED_DIM,
            norm_divisor=RELIC_ID_NORM_DIVISOR,
            expected_size=len(RELICS),
        )

    @staticmethod
    def parse(names: Sequence[str], embed_dim: int, norm_divisor: int) -> "RelicVocab":
        base = CatalogVocab.parse(names, embed_dim, norm_divisor, len(RELICS))
        return RelicVocab(
            base.names, base.embed_dim, base.norm_divisor, base.expected_size
        )

    def embed_row(self, relic_id: RelicId) -> int:
        """Row of a live relic. Never :data:`PAD_ROW`."""
        return self.names.index(relic_id.name) + 1


@dataclass(frozen=True)
class PotionVocab(CatalogVocab):
    """The potion catalog a checkpoint stores beside its weights."""

    @staticmethod
    def live() -> "PotionVocab":
        return PotionVocab(
            names=POTION_IDS,
            embed_dim=CARD_EMBED_DIM,
            norm_divisor=POTION_ID_NORM_DIVISOR,
            expected_size=len(POTION_IDS),
        )

    @staticmethod
    def parse(names: Sequence[str], embed_dim: int, norm_divisor: int) -> "PotionVocab":
        base = CatalogVocab.parse(names, embed_dim, norm_divisor, len(POTION_IDS))
        return PotionVocab(
            base.names, base.embed_dim, base.norm_divisor, base.expected_size
        )

    def embed_row(self, potion_id: str) -> int:
        """Row of a registered potion id. Never :data:`PAD_ROW`."""
        return self.names.index(potion_id) + 1


@dataclass(frozen=True)
class BossVocab(CatalogVocab):
    """The act boss catalog, ``encounters.catalog.BOSS_IDS``."""

    @staticmethod
    def live() -> "BossVocab":
        return BossVocab(
            names=BOSS_IDS,
            embed_dim=CARD_EMBED_DIM,
            norm_divisor=BOSS_ID_NORM_DIVISOR,
            expected_size=len(BOSS_IDS),
        )

    @staticmethod
    def parse(names: Sequence[str], embed_dim: int, norm_divisor: int) -> "BossVocab":
        base = CatalogVocab.parse(names, embed_dim, norm_divisor, len(BOSS_IDS))
        return BossVocab(base.names, base.embed_dim, base.norm_divisor, base.expected_size)

    def embed_row(self, boss_id: str) -> int:
        """Row of a catalog boss. Never :data:`PAD_ROW`."""
        return self.names.index(boss_id) + 1


@dataclass(frozen=True)
class MonsterVocab(CatalogVocab):
    """The enemy catalog, ``monsters.catalog.MONSTER_IDS``."""

    @staticmethod
    def live() -> "MonsterVocab":
        return MonsterVocab(
            names=MONSTER_IDS,
            embed_dim=CARD_EMBED_DIM,
            norm_divisor=MONSTER_ID_NORM_DIVISOR,
            expected_size=len(MONSTER_IDS),
        )

    @staticmethod
    def parse(names: Sequence[str], embed_dim: int, norm_divisor: int) -> "MonsterVocab":
        base = CatalogVocab.parse(names, embed_dim, norm_divisor, len(MONSTER_IDS))
        return MonsterVocab(
            base.names, base.embed_dim, base.norm_divisor, base.expected_size
        )

    def embed_row(self, monster_id: str) -> int:
        """Row of a catalog monster. Never :data:`PAD_ROW`."""
        return self.names.index(monster_id) + 1


@dataclass(frozen=True)
class RunEmbedPin:
    """The pin a run zip stores beside its weights."""

    card: CardVocab
    relic: RelicVocab
    potion: PotionVocab
    boss: BossVocab
    monster: MonsterVocab

    @staticmethod
    def live() -> "RunEmbedPin":
        return RunEmbedPin(
            CardVocab.live(),
            RelicVocab.live(),
            PotionVocab.live(),
            BossVocab.live(),
            MonsterVocab.live(),
        )

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "card_id_names": self.card.names,
            "card_embed_dim": self.card.embed_dim,
            "card_id_norm_divisor": self.card.norm_divisor,
            "relic_id_names": self.relic.names,
            "relic_id_norm_divisor": self.relic.norm_divisor,
            "potion_id_names": self.potion.names,
            "potion_id_norm_divisor": self.potion.norm_divisor,
            "boss_id_names": self.boss.names,
            "boss_id_norm_divisor": self.boss.norm_divisor,
            "monster_id_names": self.monster.names,
            "monster_id_norm_divisor": self.monster.norm_divisor,
        }

    def assert_match(self, live: "RunEmbedPin") -> None:
        """Raise :class:`CardVocabMismatch` on the first catalog that differs."""
        self.card.assert_match(live.card)
        self.relic.assert_match(live.relic)
        self.potion.assert_match(live.potion)
        self.boss.assert_match(live.boss)
        self.monster.assert_match(live.monster)

    @staticmethod
    def parse_kwargs(stored_kwargs: dict[str, Any]) -> "RunEmbedPin":
        """Refuse a zip that omitted any catalog key."""
        missing = {key for key in _PIN_KEYS if key not in stored_kwargs}
        if missing:
            raise CardVocabMismatch(f"the pin is missing {', '.join(sorted(missing))}")
        dim = int(stored_kwargs["card_embed_dim"])
        return RunEmbedPin(
            card=CardVocab.parse(
                stored_kwargs["card_id_names"],
                dim,
                int(stored_kwargs["card_id_norm_divisor"]),
            ),
            relic=RelicVocab.parse(
                stored_kwargs["relic_id_names"],
                dim,
                int(stored_kwargs["relic_id_norm_divisor"]),
            ),
            potion=PotionVocab.parse(
                stored_kwargs["potion_id_names"],
                dim,
                int(stored_kwargs["potion_id_norm_divisor"]),
            ),
            boss=BossVocab.parse(
                stored_kwargs["boss_id_names"],
                dim,
                int(stored_kwargs["boss_id_norm_divisor"]),
            ),
            monster=MonsterVocab.parse(
                stored_kwargs["monster_id_names"],
                dim,
                int(stored_kwargs["monster_id_norm_divisor"]),
            ),
        )


@dataclass(frozen=True)
class EmbedLayout:
    """Which observation indices the extractor replaces, and what it keeps.

    ``hand_id``, ``option_id``, ``owned_relic``, ``belt_potion``, ``act_boss``,
    and ``monster_id`` leave the kept block because each becomes a
    ``CARD_EMBED_DIM``-wide row. ``pile_id`` and ``pile_upgrade`` leave it
    because each pile pools into a sum and a mean of ``CARD_EMBED_DIM + 1``
    floats. ``option_kind`` stays: a kind is a small integer the MLP reads
    directly. Pile counts and sizes stay too.

    ``pile_id`` and ``pile_upgrade`` are pile-major: ``len(PILES)`` runs of
    ``PILE_SLOTS`` indices, in :data:`PILES` order.
    """

    hand_id: tuple[int, ...]
    option_kind: tuple[int, ...]
    option_id: tuple[int, ...]
    owned_relic: tuple[int, ...]
    belt_potion: tuple[int, ...]
    act_boss: tuple[int, ...]
    monster_id: tuple[int, ...]
    pile_id: tuple[int, ...]
    pile_upgrade: tuple[int, ...]
    keep: tuple[int, ...]
    embed_dim: int

    @staticmethod
    def from_run_constants(embed_dim: int) -> "EmbedLayout":
        hand_id = tuple(
            HAND_START + i * CARD_FEATURES + HAND_ID_FIELD for i in range(MAX_HAND_SIZE)
        )
        option_start = LAYOUT["options"].start
        row_size = OPTION_BLOCK.option_row_size
        option_kind = tuple(
            option_start + row * row_size + _OPTION_KIND_FIELD
            for row in range(OPTION_BLOCK.option_rows)
        )
        option_id = tuple(
            option_start + row * row_size + _OPTION_ID_FIELD
            for row in range(OPTION_BLOCK.option_rows)
        )
        owned_relic = _indices(LAYOUT.slice("owned_relics"))
        belt_potion = _indices(LAYOUT.slice("belt"))
        act_boss = _indices(LAYOUT.slice("act_boss"))
        enemies_start = LAYOUT["enemies"].start
        monster_id = tuple(
            enemies_start + slot * ENEMY_SLOT_FIELDS + ENEMY_ID_FIELD
            for slot in range(MAX_ENEMIES)
        )
        pile_id = tuple(
            LAYOUT[pile].start + slot for pile in PILES for slot in range(PILE_SLOTS)
        )
        pile_upgrade = tuple(index + PILE_SLOTS for index in pile_id)
        embedded = (
            set(hand_id)
            | set(option_id)
            | set(owned_relic)
            | set(belt_potion)
            | set(act_boss)
            | set(monster_id)
            | set(pile_id)
            | set(pile_upgrade)
        )
        keep = tuple(index for index in range(RUN_OBS_SIZE) if index not in embedded)
        return EmbedLayout(
            hand_id=hand_id,
            option_kind=option_kind,
            option_id=option_id,
            owned_relic=owned_relic,
            belt_potion=belt_potion,
            act_boss=act_boss,
            monster_id=monster_id,
            pile_id=pile_id,
            pile_upgrade=pile_upgrade,
            keep=keep,
            embed_dim=embed_dim,
        )

    @staticmethod
    def from_teacher_constants(embed_dim: int) -> "EmbedLayout":
        """Offsets a Box(718) zip stored. No act boss, monsters, or piles."""
        hand_id = tuple(
            HAND_START + i * CARD_FEATURES + HAND_ID_FIELD for i in range(MAX_HAND_SIZE)
        )
        option_start = LAYOUT_V1["options"].start
        row_size = OPTION_BLOCK.option_row_size
        option_kind = tuple(
            option_start + row * row_size + _OPTION_KIND_FIELD
            for row in range(OPTION_BLOCK.option_rows)
        )
        option_id = tuple(
            option_start + row * row_size + _OPTION_ID_FIELD
            for row in range(OPTION_BLOCK.option_rows)
        )
        owned_relic = _indices(LAYOUT_V1.slice("owned_relics"))
        belt_potion = _indices(LAYOUT_V1.slice("belt"))
        embedded = (
            set(hand_id) | set(option_id) | set(owned_relic) | set(belt_potion)
        )
        keep = tuple(index for index in range(RUN_OBS_SIZE_V1) if index not in embedded)
        return EmbedLayout(
            hand_id=hand_id,
            option_kind=option_kind,
            option_id=option_id,
            owned_relic=owned_relic,
            belt_potion=belt_potion,
            act_boss=(),
            monster_id=(),
            pile_id=(),
            pile_upgrade=(),
            keep=keep,
            embed_dim=embed_dim,
        )

    @property
    def features_dim(self) -> int:
        n_rows = (
            len(self.hand_id)
            + len(self.option_id)
            + len(self.owned_relic)
            + len(self.belt_potion)
            + len(self.act_boss)
            + len(self.monster_id)
        )
        n_piles = len(self.pile_id) // PILE_SLOTS if PILE_SLOTS else 0
        pooled = n_piles * 2 * (self.embed_dim + 1)
        return len(self.keep) + n_rows * self.embed_dim + pooled


def _indices(segment: slice) -> tuple[int, ...]:
    return tuple(range(segment.start, segment.stop))


def is_run_observation_space(space: gymnasium.Space) -> bool:
    """True only for the run ``Box(RUN_OBS_SIZE,)`` of ``float32``."""
    return (
        isinstance(space, gymnasium.spaces.Box)
        and space.shape == (RUN_OBS_SIZE,)
        and space.dtype == np.float32
    )


def is_teacher_observation_space(space: gymnasium.Space) -> bool:
    """True only for the teacher ``Box(718,)`` of ``float32``."""
    return (
        isinstance(space, gymnasium.spaces.Box)
        and space.shape == (RUN_OBS_SIZE_V1,)
        and space.dtype == np.float32
    )


def card_embed_policy_kwargs(
    net_arch: dict[str, list[int]] | None,
) -> dict[str, Any]:
    """Kwargs splat for ``MaskablePPO`` on a run env. Always carries the pin."""
    policy_kwargs: dict[str, Any] = {
        "features_extractor_class": CardEmbedExtractor,
        "features_extractor_kwargs": RunEmbedPin.live().as_kwargs(),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch
    return {"policy_kwargs": policy_kwargs}


def stored_observation_space(path: Path) -> gymnasium.Space | None:
    """The observation space an SB3 zip was saved with, if it stored one.

    A caller compares this before the card pin so an older 151 or 201 wide zip
    reports the width it really has, instead of a missing table.
    """
    stored = _read_zip_data(Path(path)).get("observation_space")
    return stored if isinstance(stored, gymnasium.Space) else None


def assert_card_vocab_match(path: Path, live: RunEmbedPin) -> None:
    """Read the zip pin. Refuse a missing key, a renamed extractor, or a catalog mismatch.

    Runs before ``MaskablePPO.load``, so a mismatched table fails without
    building a network and without touching weights. Nothing here pads,
    remaps, or falls back to a flat extractor.
    """
    stored = _read_policy_kwargs(Path(path))
    if stored is None:
        raise CheckpointVocabMismatch(path, "no policy_kwargs in the zip")

    extractor_class = stored.get("features_extractor_class")
    if extractor_class is not CardEmbedExtractor:
        raise CheckpointVocabMismatch(
            path,
            f"features_extractor_class is {extractor_class!r}, "
            f"expected {CardEmbedExtractor!r}",
        )

    stored_kwargs = stored.get("features_extractor_kwargs") or {}
    missing = {key for key in _PIN_KEYS if key not in stored_kwargs}
    if missing:
        raise CheckpointVocabMismatch(
            path, f"the pin is missing {', '.join(sorted(missing))}"
        )

    try:
        RunEmbedPin.parse_kwargs(stored_kwargs).assert_match(live)
    except (CardVocabMismatch, TypeError, ValueError) as exc:
        if isinstance(exc, CheckpointVocabMismatch):
            raise
        raise CheckpointVocabMismatch(path, str(exc)) from exc


def assert_teacher_vocab_match(path: Path, live: RunEmbedPin) -> None:
    """Card, relic, and potion pins only. A 718 zip has no boss or monster table."""
    stored = _read_policy_kwargs(Path(path))
    if stored is None:
        raise CheckpointVocabMismatch(path, "no policy_kwargs in the zip")

    extractor_class = stored.get("features_extractor_class")
    if extractor_class is not CardEmbedExtractor:
        raise CheckpointVocabMismatch(
            path,
            f"features_extractor_class is {extractor_class!r}, "
            f"expected {CardEmbedExtractor!r}",
        )

    stored_kwargs = stored.get("features_extractor_kwargs") or {}
    missing = {key for key in _PIN_KEYS_TEACHER if key not in stored_kwargs}
    if missing:
        raise CheckpointVocabMismatch(
            path, f"the teacher pin is missing {', '.join(sorted(missing))}"
        )

    try:
        card = CardVocab.parse(
            stored_kwargs["card_id_names"],
            int(stored_kwargs["card_embed_dim"]),
            int(stored_kwargs["card_id_norm_divisor"]),
        )
        relic = RelicVocab.parse(
            stored_kwargs["relic_id_names"],
            int(stored_kwargs["card_embed_dim"]),
            int(stored_kwargs["relic_id_norm_divisor"]),
        )
        potion = PotionVocab.parse(
            stored_kwargs["potion_id_names"],
            int(stored_kwargs["card_embed_dim"]),
            int(stored_kwargs["potion_id_norm_divisor"]),
        )
        card.assert_match(live.card)
        relic.assert_match(live.relic)
        potion.assert_match(live.potion)
    except (CardVocabMismatch, TypeError, ValueError) as exc:
        if isinstance(exc, CheckpointVocabMismatch):
            raise
        raise CheckpointVocabMismatch(path, str(exc)) from exc


def _read_policy_kwargs(path: Path) -> dict[str, Any] | None:
    """The zip's ``policy_kwargs``, or ``None`` when it stored none."""
    stored = _read_zip_data(Path(path)).get("policy_kwargs")
    return stored if isinstance(stored, dict) else None


def _read_zip_data(path: Path) -> dict[str, Any]:
    """``data`` from an SB3 zip, with every ``:serialized:`` value unpickled.

    SB3's own ``json_to_data`` turns a deserialize failure into a warning and
    drops the key, which would report a renamed extractor as a missing pin.
    Unpickling here keeps that a refusal with the real reason.
    """
    import base64

    import cloudpickle

    try:
        with zipfile.ZipFile(path) as archive:
            raw = json.loads(archive.read("data").decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise CheckpointVocabMismatch(path, f"unreadable SB3 zip: {exc}") from exc

    data: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and ":serialized:" in value:
            try:
                data[key] = cloudpickle.loads(base64.b64decode(value[":serialized:"]))
            except (ModuleNotFoundError, AttributeError, ImportError, TypeError) as exc:
                raise CheckpointVocabMismatch(
                    path, f"cannot rebuild saved {key}: {exc}"
                ) from exc
        else:
            data[key] = value
    return data


class CardEmbedExtractor(BaseFeaturesExtractor):
    """Replace every catalog float with a learned row, and pool each card bag.

    The constructor takes the pin as required arguments, so a zip can only
    rebuild this class from a pin it stored. Decode happens here and nowhere
    else: the env keeps writing floats, and no caller reads an embedding row.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        card_id_names: tuple[str, ...],
        card_embed_dim: int,
        card_id_norm_divisor: int,
        relic_id_names: tuple[str, ...],
        relic_id_norm_divisor: int,
        potion_id_names: tuple[str, ...],
        potion_id_norm_divisor: int,
        boss_id_names: tuple[str, ...] | None = None,
        boss_id_norm_divisor: int | None = None,
        monster_id_names: tuple[str, ...] | None = None,
        monster_id_norm_divisor: int | None = None,
    ) -> None:
        teacher = is_teacher_observation_space(observation_space)
        if not teacher and not is_run_observation_space(observation_space):
            raise CardVocabMismatch(
                "CardEmbedExtractor reads the run Box"
                f"({RUN_OBS_SIZE},) or the teacher Box({RUN_OBS_SIZE_V1},), "
                f"got {observation_space}"
            )
        card = CardVocab.parse(card_id_names, card_embed_dim, card_id_norm_divisor)
        relic = RelicVocab.parse(relic_id_names, card_embed_dim, relic_id_norm_divisor)
        potion = PotionVocab.parse(potion_id_names, card_embed_dim, potion_id_norm_divisor)
        live = RunEmbedPin.live()
        card.assert_match(live.card)
        relic.assert_match(live.relic)
        potion.assert_match(live.potion)
        if teacher:
            self._layout = EmbedLayout.from_teacher_constants(card.embed_dim)
            self._boss_divisor = 1
            self._monster_divisor = 1
            self._n_bosses = 0
            self._n_monsters = 0
        else:
            if boss_id_names is None or monster_id_names is None:
                raise CardVocabMismatch(
                    "the v2 extractor needs boss_id_names and monster_id_names"
                )
            pin = RunEmbedPin.parse_kwargs(
                {
                    "card_id_names": card_id_names,
                    "card_embed_dim": card_embed_dim,
                    "card_id_norm_divisor": card_id_norm_divisor,
                    "relic_id_names": relic_id_names,
                    "relic_id_norm_divisor": relic_id_norm_divisor,
                    "potion_id_names": potion_id_names,
                    "potion_id_norm_divisor": potion_id_norm_divisor,
                    "boss_id_names": boss_id_names,
                    "boss_id_norm_divisor": boss_id_norm_divisor,
                    "monster_id_names": monster_id_names,
                    "monster_id_norm_divisor": monster_id_norm_divisor,
                }
            )
            pin.assert_match(live)
            self._layout = EmbedLayout.from_run_constants(pin.card.embed_dim)
            self._boss_divisor = pin.boss.norm_divisor
            self._monster_divisor = pin.monster.norm_divisor
            self._n_bosses = len(pin.boss.names)
            self._n_monsters = len(pin.monster.names)

        self._card_divisor = card.norm_divisor
        self._relic_divisor = relic.norm_divisor
        self._potion_divisor = potion.norm_divisor
        self._n_cards = len(card.names)
        self._n_relics = len(relic.names)
        self._n_potions = len(potion.names)
        super().__init__(observation_space, self._layout.features_dim)

        self.card_embedding = nn.Embedding(
            self._n_cards + 1, card.embed_dim, padding_idx=PAD_ROW
        )
        self.relic_embedding = nn.Embedding(
            self._n_relics + 1, card.embed_dim, padding_idx=PAD_ROW
        )
        self.potion_embedding = nn.Embedding(
            self._n_potions + 1, card.embed_dim, padding_idx=PAD_ROW
        )
        if not teacher:
            self.boss_embedding = nn.Embedding(
                self._n_bosses + 1, card.embed_dim, padding_idx=PAD_ROW
            )
            self.monster_embedding = nn.Embedding(
                self._n_monsters + 1, card.embed_dim, padding_idx=PAD_ROW
            )
        self.register_buffer(
            "_card_kind_mask",
            torch.tensor([kind in CARD_KINDS for kind in OptionKind], dtype=torch.bool),
        )
        self.register_buffer(
            "_relic_kind_mask",
            torch.tensor([kind in RELIC_KINDS for kind in OptionKind], dtype=torch.bool),
        )
        self.register_buffer(
            "_potion_kind_mask",
            torch.tensor([kind in POTION_KINDS for kind in OptionKind], dtype=torch.bool),
        )
        self.register_buffer("_keep_index", torch.tensor(self._layout.keep, dtype=torch.long))
        self.register_buffer(
            "_hand_id_index", torch.tensor(self._layout.hand_id, dtype=torch.long)
        )
        self.register_buffer(
            "_option_kind_index", torch.tensor(self._layout.option_kind, dtype=torch.long)
        )
        self.register_buffer(
            "_option_id_index", torch.tensor(self._layout.option_id, dtype=torch.long)
        )
        self.register_buffer(
            "_owned_relic_index", torch.tensor(self._layout.owned_relic, dtype=torch.long)
        )
        self.register_buffer(
            "_belt_potion_index", torch.tensor(self._layout.belt_potion, dtype=torch.long)
        )
        if not teacher:
            self.register_buffer(
                "_act_boss_index", torch.tensor(self._layout.act_boss, dtype=torch.long)
            )
            self.register_buffer(
                "_monster_id_index", torch.tensor(self._layout.monster_id, dtype=torch.long)
            )
            self.register_buffer(
                "_pile_id_index", torch.tensor(self._layout.pile_id, dtype=torch.long)
            )
            self.register_buffer(
                "_pile_upgrade_index",
                torch.tensor(self._layout.pile_upgrade, dtype=torch.long),
            )

    def _decode_rows(
        self, norm: torch.Tensor, *, divisor: int, n_ids: int
    ) -> torch.Tensor:
        """0.0 and an out-of-range rounded index become PAD_ROW. Else index+1."""
        index = torch.round(norm * divisor) - 1
        legal = (index >= 0) & (index < n_ids)
        rows = index.to(torch.long) + 1
        return torch.where(legal, rows, torch.full_like(rows, PAD_ROW))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """``[B, RUN_OBS_SIZE]`` floats to ``[B, features_dim]``."""
        keep = observations[:, self._keep_index]

        hand_norm = observations[:, self._hand_id_index]
        hand_emb = self.card_embedding(
            self._decode_rows(hand_norm, divisor=self._card_divisor, n_ids=self._n_cards)
        )

        option_norm = observations[:, self._option_id_index]
        kinds = _decode_kinds(observations[:, self._option_kind_index])
        card_mask = self._card_kind_mask.to(kinds.device)[kinds]
        relic_mask = self._relic_kind_mask.to(kinds.device)[kinds]
        potion_mask = self._potion_kind_mask.to(kinds.device)[kinds]

        card_emb = self.card_embedding(
            self._decode_rows(option_norm, divisor=self._card_divisor, n_ids=self._n_cards)
        )
        relic_emb = self.relic_embedding(
            self._decode_rows(
                option_norm, divisor=self._relic_divisor, n_ids=self._n_relics
            )
        )
        potion_emb = self.potion_embedding(
            self._decode_rows(
                option_norm, divisor=self._potion_divisor, n_ids=self._n_potions
            )
        )
        scalar = torch.zeros_like(card_emb)
        scalar[..., 0] = option_norm
        option_emb = torch.where(card_mask.unsqueeze(-1), card_emb, scalar)
        option_emb = torch.where(relic_mask.unsqueeze(-1), relic_emb, option_emb)
        option_emb = torch.where(potion_mask.unsqueeze(-1), potion_emb, option_emb)

        owned_norm = observations[:, self._owned_relic_index]
        owned_emb = self.relic_embedding(
            self._decode_rows(
                owned_norm, divisor=self._relic_divisor, n_ids=self._n_relics
            )
        )
        belt_norm = observations[:, self._belt_potion_index]
        belt_emb = self.potion_embedding(
            self._decode_rows(
                belt_norm, divisor=self._potion_divisor, n_ids=self._n_potions
            )
        )
        parts = (
            keep,
            hand_emb.flatten(1),
            option_emb.flatten(1),
            owned_emb.flatten(1),
            belt_emb.flatten(1),
        )
        if not self._layout.act_boss:
            return torch.cat(parts, dim=1)
        boss_emb = self.boss_embedding(
            self._decode_rows(
                observations[:, self._act_boss_index],
                divisor=self._boss_divisor,
                n_ids=self._n_bosses,
            )
        )
        monster_emb = self.monster_embedding(
            self._decode_rows(
                observations[:, self._monster_id_index],
                divisor=self._monster_divisor,
                n_ids=self._n_monsters,
            )
        )
        return torch.cat(
            (
                *parts,
                boss_emb.flatten(1),
                monster_emb.flatten(1),
                self._pool_piles(observations),
            ),
            dim=1,
        )

    def _pool_piles(self, observations: torch.Tensor) -> torch.Tensor:
        """Per pile, the sum and the mean of ``[card row, upgrade bit]`` tokens."""
        batch = observations.shape[0]
        rows = self._decode_rows(
            observations[:, self._pile_id_index],
            divisor=self._card_divisor,
            n_ids=self._n_cards,
        ).view(batch, len(PILES), PILE_SLOTS)
        upgraded = observations[:, self._pile_upgrade_index].view(batch, len(PILES), PILE_SLOTS)
        occupied = (rows != PAD_ROW).to(observations.dtype).unsqueeze(-1)
        tokens = torch.cat((self.card_embedding(rows), upgraded.unsqueeze(-1)), dim=-1)
        total = (tokens * occupied).sum(dim=2)
        mean = total / occupied.sum(dim=2).clamp(min=1.0)
        return torch.cat((total, mean), dim=-1).flatten(1)


def _decode_kinds(kind_norm: torch.Tensor) -> torch.Tensor:
    """Recover the ``OptionKind`` a row wrote, the inverse of ``kind_norm``."""
    index = torch.round(kind_norm * len(OptionKind)).to(torch.long)
    return index.clamp(0, len(OptionKind) - 1)
