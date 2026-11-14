#!/usr/bin/env python3
"""Frozen Act1 RunEnv eval (box ops). Protocol: docs/act1_runenv_eval_protocol.md

Seeds 200000..200049 (50; ``--n`` extends the range, hang bar ``--n 100``).
Primary metric: act1_clear_rate (max act >= 1).
Never conflate with combat suite win rates.

Policies
--------
* ``random``: legal random among ``action_masks()==1``.
* ``model``: RunEnv-sized MaskablePPO only (obs == RUN_OBS_SIZE). Combat zips
  are rejected.
* ``hierarchical``: combat steps use the hung combat zip on ``--model``
  (alias ``--combat-model``; obs_v1 / OBS_SIZE=181) via
  ``encode_observation(CombatState)`` + combat ``get_action_mask``.
  ``--combat-policy ppo`` (default) is the hung path. ``--combat-policy jev``
  is **experimental** (HOLD failed twice on ``ecd0073``; not hang / not next
  mainline): legal-shortlist Choice, fail-open to bh_v1. See
  ``docs/COMBAT_JEV_HOLD_FAIL.md``. Non-combat default (``--jev off``): legal
  random, Jev shadow only (no action change). ``--jev on`` calls TypeSafe/Jev
  Choice. Non-combat Jev is independent of ``--combat-policy``. MAP low-HP
  ``--map-lowhp`` default on (v1 uncertain filter). ``--map-lowhp-hard``
  default **off**.

Never feed RunEnv observations into the combat model.

Hung Surplus zip:
``/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip``
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sts2_env.eval.act1_metrics import (
    _summarize,
    build_report,
    summarize_act1_rows,
    write_report,
)
from sts2_env.eval.act1_runner import (
    _legal_random,
    _run_episode,
    _run_manager,
    _selected_combat_owner,
    choose_action,
    choose_hierarchical_action,
    jev_shadow_fields,
    load_maskable_ppo,
    load_policy_models,
    model_obs_dim,
    require_obs_dim,
    validate_policy_args,
)
from sts2_env.eval.act1_suite import (
    HUNG_COMBAT_ZIP,
    JEV_SHADOW_SKIPPED,
    JEV_SHADOW_STUB,
    PROTOCOL_ID,
    SEED_COUNT,
    SEED_START,
    SEEDS,
)
from sts2_env.eval.combat_jev import CombatJevTelemetry
from sts2_env.eval.jev_client import build_jev_adapter
from sts2_env.eval.jev_config import DEFAULT_JEV_FLAGS
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, STS2RunEnv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Frozen Act1 RunEnv eval (seeds 200000..200049)")
    ap.add_argument(
        "--policy",
        choices=["random", "model", "hierarchical"],
        default="random",
        help="random | RunEnv MaskablePPO | combat zip in combat + Jev/random outside",
    )
    ap.add_argument(
        "--model",
        default="",
        help=(
            "Zip path. --policy model: RunEnv MaskablePPO (obs=RUN_OBS_SIZE). "
            "--policy hierarchical: hung combat zip (obs_v1 / OBS_SIZE=181), "
            f"e.g. {HUNG_COMBAT_ZIP}"
        ),
    )
    ap.add_argument(
        "--combat-model",
        default="",
        help="Alias of hierarchical --model (combat obs_v1 zip)",
    )
    ap.add_argument(
        "--jev",
        choices=["on", "off"],
        default="off",
        help="hierarchical non-combat: off=legal random (default); on=TypeSafe/Jev Choice",
    )
    ap.add_argument(
        "--strategic",
        choices=["jev"],
        default=None,
        help="Alias for --jev on when set to 'jev'",
    )
    ap.add_argument(
        "--jev-event",
        choices=["on", "off"],
        default="off",
        help="hierarchical + --jev on: EVENT Choice (default off; preserves MAP/REST/CARD tables)",
    )
    ap.add_argument(
        "--jev-phases",
        default="map,rest,card",
        help="Comma phases for Jev (default map,rest,card). Include event to enable EVENT",
    )
    ap.add_argument(
        "--jev-neow",
        choices=["on", "off"],
        default="off",
        help=(
            "Neow opening boon via Jev (neow_boon @ 0.65). Default off = random "
            "boon (hang: --start-with-neow + --jev-neow off, reason "
            "neow_jev_off_random). Set on for optional A/B even when "
            "--jev-event off"
        ),
    )
    ap.add_argument(
        "--map-lowhp",
        choices=["on", "off"],
        default="on",
        help=(
            "MAP low-HP v1 uncertain filter (hang default on). When hp_pressure>=2.0 "
            "and shop/rest is legal, low-confidence/error Jev resamples among safe nodes "
            "(map_lowhp_random). Does not override confident Choice. Set off to disable."
        ),
    )
    ap.add_argument(
        "--map-lowhp-hard",
        choices=["on", "off"],
        default="off",
        help=(
            "MAP low-HP v2 hard-select (default off; opt-in only). When hp_pressure>=2.0 "
            "and shop/rest is legal, always executes rest-then-shop (map_lowhp_hard), "
            "even if Jev Choice is confident. Frozen by Lab after regression."
        ),
    )
    ap.add_argument(
        "--map-lowhp-soft-b",
        choices=["on", "off"],
        default="off",
        help=(
            "MAP low-HP soft-B bias (default off; opt-in only). When hp_pressure>=2.0 "
            "and an elite/Boss is ahead on the fork, soft-prefers safer non-elite/safe options "
            "on uncertain/error (map_lowhp_soft_b). Set on to enable."
        ),
    )
    ap.add_argument(
        "--combat-policy",
        choices=["ppo", "jev", "jev-turn"],
        default="ppo",
        help=(
            "Combat source for hierarchical. Default ppo = hung bh_v1 zip "
            "(unchanged hang path). jev = experimental failed stepwise bypass. "
            "jev-turn = combat_turn_plan Choice on plan_id (enumerated steps)."
        ),
    )
    ap.add_argument(
        "--turn-plan-choice-cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "jev-turn: max plan_id Choice options (default 32, env "
            "STS2_TURN_PLAN_CHOICE_CAP, platform max 255)."
        ),
    )
    ap.add_argument(
        "--n",
        type=int,
        default=SEED_COUNT,
        help=(
            f"Episode count from seed {SEED_START} (default {SEED_COUNT}; "
            "hang bar uses --n 100)"
        ),
    )
    ap.add_argument(
        "--start-with-neow",
        action="store_true",
        default=False,
        help=(
            "Start each episode on the Neow boon screen. Hang protocol includes "
            "this flag with --jev-neow off (random boon). Default off omits Neow"
        ),
    )
    ap.add_argument(
        "--out",
        default="/workspace/sts2-sim/evals/act1_runenv_latest.json",
    )
    ap.add_argument("--max-steps", type=int, default=2000)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cap_arg = getattr(args, "turn_plan_choice_cap", None)
    if cap_arg is not None:
        from sts2_env.eval.combat_turn_plan import (
            TURN_PLAN_CHOICE_CAP_ENV,
            resolve_turn_plan_choice_cap,
        )

        os.environ[TURN_PLAN_CHOICE_CAP_ENV] = str(
            resolve_turn_plan_choice_cap(cap_arg)
        )
    validate_policy_args(args)
    model, combat_model = load_policy_models(args)
    jev_enabled = args.policy == "hierarchical" and args.jev == "on"
    jev_adapter = build_jev_adapter(enabled=jev_enabled) if args.policy == "hierarchical" else None
    jev_flags = getattr(args, "jev_flags", None) or DEFAULT_JEV_FLAGS
    combat_policy = str(getattr(args, "combat_policy", "ppo") or "ppo")
    combat_jev_telemetry = CombatJevTelemetry()
    combat_jev_adapter = None
    if args.policy == "hierarchical" and combat_policy in ("jev", "jev-turn"):
        combat_jev_adapter = build_jev_adapter(enabled=True)

    env = STS2RunEnv(character_id="Ironclad", ascension_level=0, max_steps=args.max_steps)
    rng = np.random.RandomState(0)
    rows: list[dict] = []
    t0 = datetime.now(timezone.utc)
    n = max(1, int(getattr(args, "n", SEED_COUNT)))
    seeds = list(range(SEED_START, SEED_START + n))
    for seed in seeds:
        rows.append(
            _run_episode(
                env,
                args.policy,
                model,
                combat_model,
                seed,
                rng,
                jev_enabled=jev_enabled,
                jev_adapter=jev_adapter,
                jev_flags=jev_flags,
                start_with_neow=bool(getattr(args, "start_with_neow", False)),
                combat_policy=combat_policy,
                combat_jev_adapter=combat_jev_adapter,
                combat_jev_telemetry=combat_jev_telemetry,
            )
        )
    env.close()
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    report = build_report(
        policy=args.policy,
        model_path=args.model,
        combat_model_path=args.combat_model,
        rows=rows,
        elapsed_s=elapsed,
        jev=args.jev if args.policy == "hierarchical" else "off",
        jev_event=args.jev_event if args.policy == "hierarchical" else "off",
        jev_phases=args.jev_phases if args.policy == "hierarchical" else "map,rest,card",
        jev_neow=args.jev_neow if args.policy == "hierarchical" else "off",
        start_with_neow=bool(getattr(args, "start_with_neow", False)),
        map_lowhp=getattr(args, "map_lowhp", "on") if args.policy == "hierarchical" else "on",
        map_lowhp_hard=getattr(args, "map_lowhp_hard", "off") if args.policy == "hierarchical" else "off",
        map_lowhp_soft_b=getattr(args, "map_lowhp_soft_b", "off") if args.policy == "hierarchical" else "off",
        seed_count=n,
        combat_policy=combat_policy if args.policy == "hierarchical" else "ppo",
        combat_jev_telemetry=combat_jev_telemetry,
    )
    out = Path(args.out)
    write_report(report, out)
    slim = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
