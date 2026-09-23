"""Act1 RunEnv eval helpers (hierarchical + Jev).

Heavy modules that pull the combat/card graph (``combat_jev``, Act1 runner,
metrics) are lazy. ``import sts2_env.eval`` and HOLD ``--help`` must not
complete ``core.combat`` while cards are still registering.
"""
from sts2_env.eval.act1_suite import (
    HUNG_COMBAT_ZIP,
    JEV_SHADOW_SKIPPED,
    JEV_SHADOW_STUB,
    PROTOCOL_ID,
    SEED_COUNT,
    SEED_START,
    SEEDS,
)
from sts2_env.eval.map_lowhp import (
    MAP_FIGHT_POINT_TYPES,
    MAP_LOWHP_HARD_ON,
    MAP_LOWHP_HARD_REASON,
    MAP_LOWHP_ON,
    MAP_LOWHP_PRESSURE,
    MAP_LOWHP_RANDOM_REASON,
    MAP_LOWHP_SAFE_REASON,
    MAP_LOWHP_SOFT_B_ON,
    MAP_LOWHP_SOFT_B_REASON,
    MAP_SAFE_POINT_TYPES,
    cand_has_elite_or_boss_ahead,
    is_map_fight_point,
    is_map_safe_point,
    map_lowhp_active,
    map_lowhp_filter,
    map_lowhp_filter_with_reason,
    map_lowhp_hard_item,
    map_lowhp_prefer,
    map_lowhp_safe_items,
    normalize_map_point_type,
    apply_map_lowhp_hard_policy,
    apply_map_lowhp_random_policy,
)
from sts2_env.eval.jev_types import (
    CARD_FIT_ASSIST_MIN,
    CARD_FIT_ASSIST_REASON,
    CARD_FIT_SCORE_CRITERIA,
    CHOICE_CONFIDENCE_MIN,
    CHOICE_EVENT,
    CHOICE_NEOW_BOON,
    CONTENT_MAP_REF,
    DEFAULT_JEV_PHASES,
    EVENT_CHOICE_INSTRUCTIONS,
    EVENT_OPTIONS_EMPTY_REASON,
    EVENT_SAFE_FALLBACK_REASON,
    HP_PRESSURE_ASSIST_REASON,
    HP_PRESSURE_CONTINUE,
    HP_PRESSURE_REST,
    HP_PRESSURE_SCORE_CRITERIA,
    JEV_EVENT_OFF_REASON,
    JEV_NEOW_OFF_REASON,
    JEV_PHASE_TOKENS,
    JevAnswer,
    JevError,
    NEOW_BOON_INSTRUCTIONS,
    NEOW_EARLY_CARD_INSTRUCTIONS,
    NEOW_OPTIONS_EMPTY_REASON,
    NON_JEV_PHASE_REASON,
    PLUS_CARD_CRITERION,
    POTION_OR_RELIC_REASON,
    POTION_OR_RELIC_SAFE_REASON,
    REST_CHOICE_MIN_CONFIDENCE,
    REST_HEAL_ASSIST_CONF,
    REST_SITE_HP_PRESSURE_CRITERIA,
    REST_SITE_INSTRUCTIONS,
    REST_SMITH_ASSIST_CONF,
    SHOP_RANDOM_REASON,
    SMITH_ASSIST_REASON,
    UNKNOWN_DEFER_CONF,
    UNKNOWN_DEFERRED_REASON,
    UNKNOWN_MAP_CRITERION,
    _parse_answer,
)
from sts2_env.eval.jev_telemetry import (
    default_shadow_fields,
    format_shadow_log,
)
from sts2_env.eval.jev_keys import (
    BOX_SECRETS_PATH,
    RECOMMENDED_N_ENVS_MAX,
    TYPESAFE_API_KEYS_ENV,
    TYPESAFE_API_KEY_ENV,
    TYPESAFE_BOX_SECRET_NUMBERED_MAX,
    TYPESAFE_NUMBERED_KEY_MAX,
    _dedupe_keys,
    _sync_primary_key_env,
    apply_box_secrets_to_environ,
    box_secret_key_names,
    key_for_worker,
    load_typesafe_api_keys,
    numbered_typesafe_key_env_names,
    parse_typesafe_keys_blob,
    typesafe_key_pool_summary,
    warn_n_envs,
)
from sts2_env.eval.jev_client import (
    CLOUDFLARE_1010_MIN_INTERVAL_S,
    HTTP_TIMEOUT_S,
    JevClient,
    LiveJevClient,
    StubJevClient,
    TYPESAFE_API_URL,
    TYPESAFE_HTTP_USER_AGENT,
    TYPESAFE_MODEL,
    build_jev_adapter,
    is_cloudflare_1010,
    is_typesafe_forbidden,
    typesafe_http_headers,
)
from sts2_env.eval.jev_fallback import (
    apply_choice_confidence,
    local_hp_pressure,
    rest_or_continue_override,
)
from sts2_env.eval.jev_config import (
    DEFAULT_JEV_FLAGS,
    JevPolicyFlags,
    parse_jev_phases,
    resolve_jev_flags,
)

# Lazy: these modules import CombatState / the card graph. Eager import here
# is a circular ImportError (core.combat ↔ cards) and breaks HOLD --help.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    name: ("sts2_env.eval.combat_jev", name)
    for name in (
        "CHOICE_COMBAT_STEP",
        "COMBAT_JEV_CONF_MIN",
        "COMBAT_JEV_FAILOPEN_REASONS",
        "COMBAT_JEV_HAND_TRUNCATE",
        "COMBAT_JEV_INSTRUCTIONS",
        "COMBAT_JEV_MONSTER_TRUNCATE",
        "COMBAT_JEV_TURN_CATASTROPHE_REASONS",
        "CombatJevTelemetry",
        "FAILOPEN_BAD_ID",
        "FAILOPEN_EMPTY_LIST",
        "FAILOPEN_ERROR",
        "FAILOPEN_ILLEGAL_PLAN",
        "FAILOPEN_LOW_CONF",
        "FAILOPEN_REPLAN_CAP",
        "FAILOPEN_TIMEOUT",
        "choose_combat_step",
        "classify_jev_error",
        "combat_action_id",
        "compress_combat_state",
        "enumerate_legal_combat_actions",
        "fail_open_local",
        "hung_ppo_local",
        "parse_combat_action_id",
        "summarize_combat_action",
        "summarize_combat_jev",
    )
}
_LAZY_ATTRS.update(
    {
        name: ("sts2_env.eval.act1_runner", name)
        for name in (
            "_legal_random",
            "_run_episode",
            "_run_manager",
            "_selected_combat_owner",
            "choose_action",
            "choose_hierarchical_action",
            "jev_shadow_fields",
            "load_maskable_ppo",
            "load_policy_models",
            "model_obs_dim",
            "require_obs_dim",
            "validate_policy_args",
        )
    }
)
_LAZY_ATTRS.update(
    {
        name: ("sts2_env.eval.act1_metrics", name)
        for name in (
            "_summarize",
            "build_report",
            "summarize_act1_rows",
            "write_report",
        )
    }
)


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    import importlib

    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
