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
  visible node, not invisible
* `ANCIENT` — ancient node when present

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

## Rest site
Typical options: `HEAL` (rest), `SMITH` (upgrade). Extra options (dig /
lift / …) only if the corresponding relic enabled them.

## Out of scope
Do not mix this file with combat-suite (bare / loadout_v0 / loadout_v1)
metrics. Hierarchical + Jev is **not** an Act1-clear gate.
