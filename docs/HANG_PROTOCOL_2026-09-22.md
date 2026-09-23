# Hang protocol (2026-09-22, post Neow A/B)

**Bar (n=100):** clear **3%** / median **6.5**  
**History:** n50 4%/8 (pre–start_with_neow hang) — do not use as current bar.

**Eval flags:** hang zip `combat_ppo_obs_v1_bh_v1` + hierarchical  
`--jev on --jev-event off --jev-neow off --start-with-neow`

(Box `--jev-mode suggest_live` is the same live Choice path as PR `--jev on`.)

**Jev:** MAP / REST / CARD on; EVENT off; Neow random (`--jev-neow` default off).  
MAP low-HP routing (`--map-lowhp`, default **on**, v1 uncertain filter only): thin HP (`hp_pressure >= 2.0`) + shop/rest legal → low-confidence/error Jev resamples among safe nodes (`map_lowhp_random`). Confident Choice is not overridden. Hard-select (`--map-lowhp-hard`, reason `map_lowhp_hard`) is **opt-in only** (default **off**; froze after n100 clear 0%). Hang protocol does not use hard. Hang zip and EVENT-off flags unchanged.

Opening Neow does **not** drag; Jev picking the boon **does**. Hang keeps the Neow screen (`--start-with-neow`) and randomizes the boon.

**Win gate:** Act1 clear ≥5% on this protocol.

**Combat HOLD (separate lock):** `docs/HOLD_PROTOCOL.md`. Dual gate is Act1 clear ≥5% **and** loadout_v1 HOLD ≥70 / Boss≥40 on that aligned protocol (fixtures 01–03, enc 16–21, fixture relics/potions). Hang zip stays `bh_v1`.

**Combat-Jev (`--combat-policy jev`):** HOLD failed twice; second on `ecd0073` (B 7.8 / 15.6 / Boss 0.0 vs A 74.2 / 98.9 / 49.4; failopen 33.8% but worse than first fail → selected steps toxic). **Not** hang / **not** next mainline. Hang combat stays `bh_v1`. Details: `docs/COMBAT_JEV_HOLD_FAIL.md`.

REST calibration code may remain; no win claim from it. Strategic claims frozen (EVENT off, neow Jev off) until the lab unfreezes.

**Secondary A (EVENT / relic knife):** evaluated with explicit `--jev-event on` (hang default remains `--jev-event off`). Closes low-conf / API error holes in EVENT and relic picks with safe fallback (`event_safe_fallback`, `potion_or_relic_safe_fallback`).

**Soft B (MAP elite/Boss low-HP soft bias):** `--map-lowhp-soft-b on|off` (default **off**, opt-in via `--map-lowhp-soft-b on`). When enabled, `hp_pressure >= 2.0` and an elite/Boss is ahead on the fork, uncertain/error decisions soft-prefer safe shop/rest nodes (`map_lowhp_soft_b`). Hard-select remains opt-in off (`--map-lowhp-hard off`). Hang tip stops at `58db7d0`/`dcc44b3` narrow lineage; expand `4c85dbd` remains abandoned. Hang mainline uses soft-B **off** by default.
