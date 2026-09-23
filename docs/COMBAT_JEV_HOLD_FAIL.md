# Combat-Jev HOLD smoke — failed (Lab lock)

**Status:** archived failed evidence (two HOLD collapses). **Not** hang. **Not** next mainline.

Combat-Jev bypass (`--combat-policy jev`, Choice `combat_step_choice`) is **experimental** only. Hang combat stays **`bh_v1`**. Default remains `--combat-policy ppo`. Strategy / event Jev surfaces unchanged. Do not delete the flag; do not treat it as hang. No train. No Act1 n100 from this path.

HOLD protocol: `docs/HOLD_PROTOCOL.md`. Hang protocol: `docs/HANG_PROTOCOL_2026-09-22.md`.

## First collapse — tip `2b1dcf4` (archive docs `189142d`)

| arm | overall | elite | Boss |
|---|---|---|---|
| A ppo (hung `bh_v1`) | **74.2** | 98.9 | **49.4** |
| B jev | 18.6 | 36.7 | 0.6 |

Δ **−55.6pp / −48.8pp** (overall / elite). `failopen_rate` **41.2%** (mostly `low_conf`). Damage was from **Jev-chosen** steps, not insufficient fail-open.

## Second collapse — tip `ecd0073` (Jev router lock **0.93**)

After tip① prompt/summary (`1da5744`) and tip② conf gate **0.45** (`ecd0073`), second HOLD smoke:

| arm | overall | elite | Boss |
|---|---|---|---|
| A ppo (hung `bh_v1`) | **74.2** | 98.9 | **49.4** |
| B jev | 7.8 | 15.6 | 0.0 |

Δ **−66.4pp overall / −49.4pp Boss** → collapse; **stop**; no Act1.

B telemetry: `jev_calls=7174`, failopen **33.8%** (mostly `low_conf`), latency p50 **357ms** / p95 **482ms**.

### Reading (Lab)

Compared to the first fail, calibration **lowered fail-open** (41.2% → 33.8%) but **win rates got worse** (B overall 18.6 → 7.8; Boss 0.6 → 0.0). The poison is **Jev-selected combat steps**, not the confidence gate alone. Abandon combat-Jev mainline again.

**Track 2 (design):** advisory `bh_assist` + turn-plan Choice — **not** stepwise `combat_step_choice`. Hang execution stays `bh_v1` until turn-plan HOLD clears. Contract: `docs/BH_ASSIST_CONTRACT.md`.
