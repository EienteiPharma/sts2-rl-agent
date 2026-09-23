# Jev non-combat wire

Hierarchical `--jev on` drives TypeSafe/Jev for **MAP / REST / CARD** only by default.
Choice confidence **≥ 0.65** (frozen). Combat zip path is unchanged; Jev never
runs in combat. Shop stays legal random.

Surface: `scripts/jev_noncombat.py` re-exports `sts2_env.eval.jev_policy`.
Eval: `scripts/eval_act1_runenv.py`.

## Default phases (`--jev on`, `--jev-event off`)

| Phase | Choice / Score | Notes |
|-------|----------------|-------|
| `MAP_CHOICE` | `map_fork` or `rest_or_continue` + `hp_pressure` | Strip `UNASSIGNED`. `UNKNOWN` is legal/visible. **map_lowhp v2** (hang default **on**): `hp_pressure>=2` + shop/rest legal → **hard-select** rest-then-shop (`map_lowhp_hard`), even if Jev Choice is confident fight. Counted in eval (`map_lowhp_hard_n`, stderr). Only-fight forks keep full-pool random. `--map-lowhp off` disables. |
| `REST_SITE` | rest-site Choice | heal / smith / relic options; pending `choose`/`confirm_choice` → combat slots (same as EVENT) |
| `CARD_REWARD` | `card_reward` Choice + `card_fit` Score | potion/relic screens do **not** call CARD Jev (`potion_or_relic_reward_random`) |
| `EVENT` | **off** | legal random, `jev_event_off_random` |
| `EVENT` (Neow) | **off** | random boon, `neow_jev_off_random`; `--jev-neow on` optional A/B |
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

## MAP low-HP (`map_lowhp`, hang default on, v2 hard-select)

When `hp_pressure >= 2.0` (same band as rest prefer; local fallback
`max_hp/hp` when Score is missing) and a legal map node is `SHOP` or
`REST_SITE`:

* **Hard-select** rest-then-shop (`map_lowhp_hard`) regardless of Jev
  confidence, Choice pick, or random. Monster / elite / boss / treasure /
  unknown are all overridden. If Choice already picked shop/rest, still
  tag so sentry can count (v1 ok-path was silent).
* If **only fight nodes** remain, keep full-pool random (documented; no
  tag). `--jev off` full-legal-random is unchanged.
* Knob: `--map-lowhp on|off` (CLI default **on**). Constant
  `MAP_LOWHP_ON` / `MAP_LOWHP_PRESSURE = 2.0` in `sts2_env/eval/jev.py`.
  Hang `--jev on` uses this path in `choose_jev_noncombat`. Eval counts
  `map_lowhp_hard_n` per episode and prints `map_lowhp_hard ...` on stderr.

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

Call Choice only when there are **≥ 2 non-Leave** legal options. Otherwise
skip (status `skipped`, **not** land-rate): reason `event_options_empty`
(non-Neow) or `neow_options_empty` (Neow). Same gate as Neow.

`RunManager._actions_event` must **not** invent Leave when `_event_model` is
set but `_event_options` is empty (reward / pending gap). Leave is only
emitted when `event_model is None`. Invented Leave on that gap was a
false Leave-only screen.

## Neow (`neow_boon`)

If the EVENT screen is Neow / boon (`event_id==Neow` or id/meta contains
`boon`), the Choice name is `neow_boon` and the shadow phase may be `NEOW`.
If Neow is **not** detected, skip `neow_boon` silently and use `event_choice`.
`--jev-neow` default **off** (independent of `--jev-event`): detected Neow
is legal random (`neow_jev_off_random`). `--jev-neow on` is optional A/B
(`neow_boon` @ global 0.65 even when `--jev-event off`).

**Hang protocol includes `--start-with-neow` with `--jev-neow off` (random
boon).** Opening Neow does not drag; Jev picking the boon does. Gym:

```python
env.reset(seed=seed, options={"start_with_neow": True})
# → RunManager(..., start_with_neow=True)
```

`RunEnv.reset` and `_enter_neow` import `sts2_env.events` **before**
`get_event("Neow")`. Card factory reads `docs/CARDS_REFERENCE.md` from the
package/repo root (not process cwd), so Neow from `cwd=/tmp` still yields
three `event_choice` boons.

If the Neow screen has **fewer than 2 non-Leave** legal options (Leave-only
stub), do **not** call Jev; reason `neow_options_empty` (not land-rate).

n=100 Leave-only was **pre** `import sts2_env.events` / package-rooted
`CARDS_REFERENCE`. Surplus re-smokes after that fix.

Neow is an opening boon, not a mid-act fixture.

## CLI

```bash
# hang (MAP/REST/CARD Jev; EVENT off; Neow screen + random boon)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event off --jev-neow off --start-with-neow

# prior tables without opening Neow
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event off

# EVENT Choice names (Neow still random unless --jev-neow on)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-event on --start-with-neow

# optional A/B: Jev picks Neow boon (neow_boon @ 0.65)
python scripts/eval_act1_runenv.py --policy hierarchical --model HUNG.zip --jev on --jev-neow on --start-with-neow
```

`--jev-phases map,rest,card,event` is equivalent to `--jev-event on`.
`--jev-neow` defaults to **off** (random boon, `neow_jev_off_random`).
Hang passes `--start-with-neow` (CLI default still off).

## Contract

`docs/JEV_EVENT_NEOW_CONTRACT.md`.

## REST calibration v1 (2026-09-22)

Global Choice ≥0.65 **unchanged** for MAP / CARD / EVENT / Neow.

REST_SITE only:
- Soft land: `REST_CHOICE_MIN_CONFIDENCE = 0.50` (replaces 0.65 on REST).
- Keep `_apply_hp_pressure_bias`.
- After bias, Score assists (parity with `card_fit`):
  - `hp_pressure≥2` + HEAL/REST + `conf≥0.30` → land, `reason=jev_hp_pressure_assist`
  - `hp_pressure≤1` + SMITH + `conf≥0.40` → land, `reason=jev_smith_assist`
- Smoke gate: REST `used≥30%` (denom = REST decisions; split `jev_suggest_live` / `jev_hp_pressure_assist` / `jev_smith_assist` / `low_confidence_random`).
- If Act1 clear/median regresses vs hang **3%/6.5** after n=100 → disable assists; keep soft 0.50 or fall back to random. (n50 4%/8 is history only.)
- REST cal may remain in code; **no win claim**.

## Neow hang default (`--jev-neow off`, 2026-09-22)

Ordinary EVENT with `--jev-event off` stays legal random (`jev_event_off_random`).
Detected Neow is **also random by default** (`neow_jev_off_random`).
`--jev-neow on` is optional A/B: Choice `neow_boon`, global land **≥ 0.65**,
even when `--jev-event off`. REST soft 0.50 / heal/smith assists stay
REST_SITE-only and **do not** apply to Neow.

Hang: `--start-with-neow --jev-neow off` (opening screen, random boon).
Combat zip and MAP/CARD 0.65 are unchanged. MAP low-HP `map_lowhp` is
**on** by default (v2 **hard-select** rest-then-shop under pressure;
`--map-lowhp off` to disable).
