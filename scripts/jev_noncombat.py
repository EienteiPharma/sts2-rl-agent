"""Box surface for non-combat Jev CARD_REWARD wiring.

Potion/relic screens share ``PHASE_CARD_REWARD`` with true ``pick_card``.
Those screens must not call card_reward Jev (legal-random,
``potion_or_relic_reward_random``) so they do not pollute CARD land-rate.

True pick_card maps by list order to ``_CARD_RWD_START+i`` / extra, skip
only when ``action==skip``, labels carry short card blurbs, and Choice
is Neow+early natural Act1 (not mid-act fixtures). Score ``card_fit``
assists a sub-0.65 Choice when fit >= 2.0 and the choice is not skip.

Implementation lives in ``sts2_env.eval``; this module is the
``scripts/jev_noncombat.py`` path the lab already patched.
"""
from __future__ import annotations

from sts2_env.eval.jev import (
    CARD_FIT_ASSIST_MIN,
    CARD_FIT_ASSIST_REASON,
    CARD_FIT_SCORE_CRITERIA,
    CHOICE_CONFIDENCE_MIN,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
)
from sts2_env.eval.jev_policy import (
    DECISION_CARD_REWARD,
    DECISION_POTION_OR_RELIC,
    POTION_OR_RELIC_ACTIONS,
    SKIP_ACTIONS,
    build_options,
    card_blurb,
    choose_jev_noncombat,
    is_potion_or_relic_reward,
)

__all__ = [
    "CARD_FIT_ASSIST_MIN",
    "CARD_FIT_ASSIST_REASON",
    "CARD_FIT_SCORE_CRITERIA",
    "CHOICE_CONFIDENCE_MIN",
    "DECISION_CARD_REWARD",
    "DECISION_POTION_OR_RELIC",
    "NEOW_EARLY_CARD_INSTRUCTIONS",
    "PLUS_CARD_CRITERION",
    "POTION_OR_RELIC_ACTIONS",
    "POTION_OR_RELIC_REASON",
    "SKIP_ACTIONS",
    "build_options",
    "card_blurb",
    "choose_jev_noncombat",
    "is_potion_or_relic_reward",
]
