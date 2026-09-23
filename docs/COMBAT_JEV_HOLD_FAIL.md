# Combat-Jev HOLD smoke — failed (Lab lock 2026-09-23)

**Status:** archived failed evidence. **Not** hang. **Not** next mainline.

Combat-Jev bypass (`--combat-policy jev`, Choice `combat_step_choice`) failed HOLD smoke on tip `2b1dcf4`:

| arm | overall | elite | Boss |
|---|---|---|---|
| A ppo (hung `bh_v1`) | **74.2** | 98.9 | **49.4** |
| B jev | 18.6 | 36.7 | 0.6 |

Δ **−55.6pp / −48.8pp** (overall / elite) → collapse. `failopen_rate` **41.2%** (mostly `low_conf`). Damage was from **Jev-chosen** steps, not insufficient fail-open.

Hang combat stays **`bh_v1`**. Default remains `--combat-policy ppo`. Strategy / event Jev surfaces unchanged. The CLI flag is kept as **experimental** / failed-bypass; do not delete it and do not treat it as hang. No train. No Act1 n100 from this path.

HOLD protocol: `docs/HOLD_PROTOCOL.md`. Hang protocol: `docs/HANG_PROTOCOL_2026-09-22.md`.
