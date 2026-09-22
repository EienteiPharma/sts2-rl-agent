# Act1 content map (Jev criteria)

Full Surplus content map is not on this tip. This file is the stable reference
string baked into hierarchical `--jev on` prompts. Expand in place; do not
fork a second map.

TODO(Surplus/Jev): replace this stub with the live content map used for
TypeSafe/Jev `TYPESAFE_API_KEY` calls. The eval loop already reads this path
via `sts2_env.eval.jev.CONTENT_MAP_REF`; do not rewrite `scripts/eval_act1_runenv.py`.

## Scope
Act1 (act index 0) Ironclad, ascension 0, frozen eval seeds 200000..200049.

## Map nodes (`map_fork` / `rest_or_continue`)
Legal visible `MapPointType` values the eval may offer:

* `MONSTER` — hallway fight
* `ELITE` — elite fight
* `BOSS` — act boss (end of act)
* `REST_SITE` — rest (heal / smith / relic options)
* `SHOP` — merchant
* `TREASURE` — chest
* `UNKNOWN` — question-mark (event / possible fight); this is a **legal**
  visible node, not invisible. Criteria: Unknown is a risk. If HP is thin,
  prefer rest or a safer fork when legal.
* `ANCIENT` — ancient node when present

When legal map options include `UNKNOWN`, still Score `hp_pressure`. If
`hp_pressure >= 2.0` and a non-Unknown legal node exists and Choice picked
Unknown with confidence **< 0.80** → defer, legal random among non-Unknown,
reason `unknown_deferred`. 0.80 is **defer-only**; Choice land threshold
stays **0.65**.

Strip `UNASSIGNED` and any action whose RunEnv `action_mask` bit is 0
before Choice. Do not invent nodes that are not on the current fork.

`rest_or_continue` applies when the current fork contains at least one
`REST_SITE` and at least one non-rest node. HP pressure Score (lab-hung):

* ≥ 2.0 → prefer rest
* ≤ 1.0 → prefer continue
* otherwise trust Choice (still require Choice confidence ≥ 0.65)

## Card rewards
Act1 combat/hallway card rewards are **not** pre-upgraded. A `+` / upgraded
card in a reward list is not a natural drop — it comes from **Smith** (rest
site) or **Neow** (opening). Do not prefer upgraded reward cards as if the
reward itself upgraded them.

Choice instructions are **Neow+early natural Act1**, not mid-act fixtures.
`PHASE_CARD_REWARD` is also used for potion and relic reward screens
(`pick_potion` / `pick_relic_reward`); those are **not** `pick_card` and
must not be scored as CARD land-rate.

`card_fit` Score (lab-hung, 4-level, mirrors rest Score):

* Choice confidence threshold stays **0.65**
* confidence < 0.65 and `card_fit >= 2.0` and choice ≠ skip → land
  (`jev_card_fit_assist`)
* otherwise uncertain → legal random

## Rest site
Typical options: `HEAL` (rest), `SMITH` (upgrade). Extra options (dig /
lift / …) only if the corresponding relic enabled them.

## Events (`event_choice`, optional `--jev-event on`)
Default eval keeps EVENT **off** (legal random, `jev_event_off_random`) so
MAP/REST/CARD tables stay comparable. When on: Choice name `event_choice`;
`event_choice` index i → `_EVENT_START+i`; pending `confirm_choice` →
`_COMBAT_START`; pending `choose` i → `_COMBAT_START+1+i`. Criteria follow
option label/description (HP, gold, cards, relics). Events marked **待核**:
**literal risks only** — do not invent hard rules.

Neow/boon screens (`event_id=Neow` or id/meta contains boon) use Choice
`neow_boon` (opening tolerance, not a mid-act fixture). If Neow is not
detected, skip `neow_boon` silently. `--jev-neow off` skips Jev on a
detected Neow screen.

Shop stays legal random.

## Out of scope
Do not mix this file with combat-suite (bare / loadout_v0 / loadout_v1)
metrics. Hierarchical + Jev is **not** an Act1-clear gate.
