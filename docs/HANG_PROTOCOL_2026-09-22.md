# Hang protocol (2026-09-22, post Neow A/B)

**Bar (n=100):** clear **3%** / median **6.5**  
**History:** n50 4%/8 (pre–start_with_neow hang) — do not use as current bar.

**Eval flags:** hang zip `combat_ppo_obs_v1_bh_v1` + hierarchical  
`--jev on --jev-event off --jev-neow off --start-with-neow`

(Box `--jev-mode suggest_live` is the same live Choice path as PR `--jev on`.)

**Jev:** MAP / REST / CARD on; EVENT off; Neow random (`--jev-neow` default off).  
MAP low-HP routing (`--map-lowhp`, default **on**): thin HP + shop/rest legal → do not random-fall to monster/elite. Hang zip and EVENT-off flags unchanged.

Opening Neow does **not** drag; Jev picking the boon **does**. Hang keeps the Neow screen (`--start-with-neow`) and randomizes the boon.

**Win gate:** Act1 clear ≥5% on this protocol.

**Combat HOLD (separate lock):** `docs/HOLD_PROTOCOL.md`. Dual gate is Act1 clear ≥5% **and** loadout_v1 HOLD ≥70 / Boss≥40 on that aligned protocol (fixtures 01–03, enc 16–21, fixture relics/potions). Hang zip stays `bh_v1`.

REST calibration code may remain; no win claim from it. Strategic claims frozen (EVENT off, neow Jev off) until the lab unfreezes.
