# Jev non-combat wire

Hierarchical `--jev on` drives TypeSafe/Jev for **MAP / REST / CARD** only by default.
Choice confidence **≥ 0.65** (frozen). Combat zip path is unchanged; Jev never
runs in combat. Shop stays legal random.

Surface: `scripts/jev_noncombat.py` re-exports `sts2_env.eval.jev_policy`.
Eval: `scripts/eval_act1_runenv.py`.

## Default phases (`--jev on`, `--jev-event off`)

| Phase | Choice / Score | Notes |
|-------|----------------|-------|
| `MAP_CHOICE` | `map_fork` or `rest_or_continue` + `hp_pressure` | Strip `UNASSIGNED`. `UNKNOWN` is legal/visible. |
| `REST_SITE` | rest-site Choice | heal / smith / relic options |
| `CARD_REWARD` | `card_reward` Choice + `card_fit` Score | potion/relic screens do **not** call CARD Jev (`potion_or_relic_reward_random`) |
| `EVENT` | **off** | legal random, `jev_event_off_random` |
| `SHOP` | **never** | legal random |

## MAP `UNKNOWN` (no new phase)

When legal map options include `point_type=UNKNOWN`:

* Criteria on that node: Unknown is a risk (event or fight); if HP is thin,
  prefer rest / a safer fork when legal. See `docs/act1_content_map.md`.
* Keep `hp_pressure` Score.
* If `hp_pressure >= 2` and a non-Unknown legal node exists and Choice picked
  Unknown with **confidence < 0.80** → defer, legal random among non-Unknown,
  reason `unknown_deferred`. Logged on the shadow record.
* 0.80 is **only** this defer; the Choice land threshold stays **0.65**.

## EVENT (`--jev-event on`)

`EVENT` joins `JEV_PHASES` only when `--jev-event on` or `--jev-phases` lists
`event`. Default off so existing MAP/REST/CARD tables stay comparable.

`build_event_options`:

* `event_choice` index `i` → `_EVENT_START + i` (mask bit 1, `enabled` only)
* pending `confirm_choice` → `_COMBAT_START`
* pending `choose` index `i` → `_COMBAT_START + 1 + i` (same as `run_env`)

TypeSafe Choice name is `event_choice`. State includes `content_map`, HP, gold,
deck size, `event_id`. Criteria follow option label/description; events marked
「待核」get **literal risks only** (no invented hard rules).

Pending choose/confirm on EVENT is **wired** (not fail-open random).

## Neow (`neow_boon`)

If the EVENT screen is Neow / boon (`event_id==Neow` or id/meta contains
`boon`), the Choice name is `neow_boon` and the shadow phase may be `NEOW`.
If Neow is **not** detected, skip `neow_boon` silently and use `event_choice`.
`--jev-neow off` skips Jev on a detected Neow screen (legal random).

**Measure `neow_boon` only with `--start-with-neow`.** Hang tables and this
round's n=100 omit Neow (`STS2RunEnv.reset` default / eval CLI default off
→ map start). Gym pass-through:

```python
env.reset(seed=seed, options={"start_with_neow": True})
# → RunManager(..., start_with_neow=True)
```

Neow is an opening boon, not a mid-act fixture.

## CLI

```bash
# prior tables (EVENT off)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event off

# EVENT(+Neow Choice names; still no opening Neow unless --start-with-neow)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event on

# Measure neow_boon (hang / n=100 tables stay without this flag)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event on --start-with-neow
```

`--jev-phases map,rest,card,event` is equivalent to `--jev-event on`.
`--jev-neow` defaults to follow `--jev-event`.
`--start-with-neow` is **off** unless measuring `neow_boon` (hang / n=100 stay without Neow).

## Contract

`docs/JEV_EVENT_NEOW_CONTRACT.md`.
