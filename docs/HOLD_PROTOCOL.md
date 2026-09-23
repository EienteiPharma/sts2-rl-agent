# loadout_v1 HOLD protocol (LOCKED 2026-09-23)

**Scope:** Ironclad combat-suite HOLD (elite+boss). **Not** Act1 RunEnv hang protocol (`docs/HANG_PROTOCOL_2026-09-22.md`).

**Hang zip:** `combat_ppo_obs_v1_bh_v1/final_model.zip` (unchanged). Do not swap until dual gate.

**Dual gate (评测哨兵):** Act1 RunEnv clear **≥5%** on hang flags **and** this HOLD overall **≥70** / Boss **≥40**. Buffer train stays frozen until Lab says otherwise. Do not invent new train fracs.

## Hang table (2026-09-22)

`evals/obs_v1_bh_v1_loadout_v1_n20.summary.json` on the hang zip:

| overall | elite | Boss | n_eps |
|---|---|---|---|
| **74.2%** (0.7417) | 98.9% (0.9889) | **49.4%** (0.4944) | 20 |

360 fights (3 fixtures × 6 enc × 20). By fixture: 01 93.3 / 02 58.3 / 03 70.8. Homogenizing 01–03 to Shuriken-only and stripping potions is **not** this table (Surplus re-verify on that rewrite: 41.1 / 76.7 / Boss 5.6).

## Locked layout

| Field | Value |
|---|---|
| Suite | `loadout_v1` |
| Fixtures | hang-era `scripts/fixtures/loadout_v1/loadout_v1_01.json` … `_03.json` (LOCKED; do not rewrite) |
| Encounters | ids **16–21** (`ALL_ACT1_ENCOUNTERS`: elite 16–18, boss 19–21) |
| Episodes | hang table **`n_eps=20`** → 3×6×20 = **360** fights. Smoke `n_eps=1` is 18 fights, not the table. |
| Seed | `40000 + fixture_index*1000 + enc_id*100 + ep` (`env.reset(seed=…)`) |
| Win | `terminated and reward > 0` (sparse +1 win / −1 loss) |
| Max gym steps | 400 |
| Relics / potions | **Exactly** the per-fixture lists below, applied into `CombatState`. Default `BURNING_BLOOD+SHURIKEN` only if a fixture **omits** the `relics` key. Never overwrite LOCKED relics/potions. |
| Reset path | Hang-era `options_from_fixture` → `STS2CombatEnv.reset(..., options={deck,hp,max_hp,relics,potions})`. `loadout_provider` also applies the same keys. |
| PYTHONPATH | Repo checkout `sts2_env` (`PYTHONPATH=.` or editable install). Do not mix a stray `/workspace/sts2-sim` copy of `eval_combat_suite.py` (bare 22-enc, no `--suite loadout_v1`). |

### LOCKED HOLD fixtures (hang-era, 2026-09-22)

| stem | hp | relics | potions | deck note |
|---|---|---|---|---|
| `loadout_v1_01` | 58 | `BURNING_BLOOD`, `SHURIKEN` | `FirePotion`, `AttackPotion`, `null` | mid_multihit (Sword Boomerang 3/3/4 + Twin×2 + Thrash) |
| `loadout_v1_02` | 60 | `BURNING_BLOOD`, `BAG_OF_MARBLES` | `ExplosiveAmpoule`, `BlockPotion`, `null` | mid_aoe (Thunderclap×2 + Whirlwind+ + Shockwave) |
| `loadout_v1_03` | 52 | `BURNING_BLOOD`, `VAJRA` | `StrengthPotion`, `FlexPotion`, `null` | mid_strength (Inflame+ / Uppercut / Anger) |

## Runners (same protocol)

Full hang table (n=20):

```bash
PYTHONPATH=. python scripts/eval_combat_suite.py \
  --suite loadout_v1 --n-eps 20 \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --out evals/obs_v1_bh_v1_loadout_v1_n20.summary.json
```

Expect overall ≈74% and Boss ≈49% on `bh_v1` (tolerance ~±3–5pp). Gate remains **≥70 / Boss≥40** on this protocol.

## Not this protocol

- Bare Act1 22-enc (`eval_combat_suite.py` without `--suite loadout_v1`).
- Act1 RunEnv hierarchical (`scripts/eval_act1_runenv.py`).
- Continue-train / mix fracs. HOLD alignment is eval plumbing, not a new recipe.
- Rewriting 01–03 to a shared Shuriken deck/hp or stripping potions.
