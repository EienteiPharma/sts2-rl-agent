"""Box surface for non-combat Jev wiring (MAP / REST / CARD + optional EVENT).

Default ``JEV_PHASES`` is map, rest, card. EVENT joins only with
``--jev-event on`` (or ``--jev-phases`` listing ``event``). Shop stays
legal random. MAP ``UNKNOWN`` keeps ``hp_pressure`` and may defer with
reason ``unknown_deferred``. EVENT pending ``choose`` / ``confirm_choice``
maps to combat slots via ``build_event_options``.

Potion/relic screens share ``PHASE_CARD_REWARD`` with true ``pick_card``.
Those screens must not call card_reward Jev (legal-random,
``potion_or_relic_reward_random``) so they do not pollute CARD land-rate.

Implementation lives in ``sts2_env.eval``; this module is the
``scripts/jev_noncombat.py`` path the lab already patched.
"""
from __future__ import annotations

from sts2_env.eval.jev import (
    CARD_FIT_ASSIST_MIN,
    CARD_FIT_ASSIST_REASON,
    CARD_FIT_SCORE_CRITERIA,
    CHOICE_CONFIDENCE_MIN,
    CHOICE_EVENT,
    CHOICE_NEOW_BOON,
    DEFAULT_JEV_PHASES,
    EVENT_CHOICE_INSTRUCTIONS,
    JEV_EVENT_OFF_REASON,
    JEV_NEOW_OFF_REASON,
    NEOW_BOON_INSTRUCTIONS,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
    SHOP_RANDOM_REASON,
    UNKNOWN_DEFER_CONF,
    UNKNOWN_DEFERRED_REASON,
    UNKNOWN_MAP_CRITERION,
)
from sts2_env.eval.jev_policy import (
    DECISION_CARD_REWARD,
    DECISION_EVENT,
    DECISION_NEOW,
    DECISION_POTION_OR_RELIC,
    DEFAULT_JEV_FLAGS,
    JevPolicyFlags,
    POTION_OR_RELIC_ACTIONS,
    SKIP_ACTIONS,
    build_event_options,
    build_options,
    card_blurb,
    choose_jev_noncombat,
    is_neow_or_boon_screen,
    is_potion_or_relic_reward,
    parse_jev_phases,
    resolve_jev_flags,
)

# Lab-facing alias: EVENT is not in the default set.
JEV_PHASES = DEFAULT_JEV_PHASES

__all__ = [
    "CARD_FIT_ASSIST_MIN",
    "CARD_FIT_ASSIST_REASON",
    "CARD_FIT_SCORE_CRITERIA",
    "CHOICE_CONFIDENCE_MIN",
    "CHOICE_EVENT",
    "CHOICE_NEOW_BOON",
    "DECISION_CARD_REWARD",
    "DECISION_EVENT",
    "DECISION_NEOW",
    "DECISION_POTION_OR_RELIC",
    "DEFAULT_JEV_FLAGS",
    "DEFAULT_JEV_PHASES",
    "EVENT_CHOICE_INSTRUCTIONS",
    "JEV_EVENT_OFF_REASON",
    "JEV_NEOW_OFF_REASON",
    "JEV_PHASES",
    "JevPolicyFlags",
    "NEOW_BOON_INSTRUCTIONS",
    "NEOW_EARLY_CARD_INSTRUCTIONS",
    "PLUS_CARD_CRITERION",
    "POTION_OR_RELIC_ACTIONS",
    "POTION_OR_RELIC_REASON",
    "SHOP_RANDOM_REASON",
    "SKIP_ACTIONS",
    "UNKNOWN_DEFER_CONF",
    "UNKNOWN_DEFERRED_REASON",
    "UNKNOWN_MAP_CRITERION",
    "build_event_options",
    "build_options",
    "card_blurb",
    "choose_jev_noncombat",
    "is_neow_or_boon_screen",
    "is_potion_or_relic_reward",
    "parse_jev_phases",
    "resolve_jev_flags",
]
