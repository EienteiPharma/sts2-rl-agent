# loadout_v1 HOLD protocol (LOCKED 2026-09-23)

**Scope:** Ironclad combat-suite HOLD (elite+boss). **Not** Act1 RunEnv hang protocol (`docs/HANG_PROTOCOL_2026-09-22.md`).

**Hang zip:** `combat_ppo_obs_v1_bh_v1/final_model.zip` (unchanged). Do not swap until dual gate.

**Dual gate (评测哨兵):** Act1 RunEnv clear **≥5%** on hang flags **and** this HOLD overall **≥70** / Boss **≥40**. Buffer train stays frozen until Lab says otherwise. Do not invent new train fracs.

## Hang table (2026-09-22)

`evals/obs_v1_bh_v1_loadout_v1_n20.summary.json` on the hang zip:

| overall | elite | Boss | n_eps |
|---|---|---|---|
| **74.2%** | 98.9% | **49.4%** | 20 |

0-step HOLD on the **same zip** via a later `run_hold_smoke` that dropped fixture relics/potions was **39.4 / 78.9 / Boss 0.0**. That path is not the hang table.

## Locked layout

| Field | Value |
|---|---|
| Suite | `loadout_v1` |
| Fixtures | `scripts/fixtures/loadout_v1/loadout_v1_01.json` … `_03.json` |
| Encounters | ids **16–21** (`ALL_ACT1_ENCOUNTERS`: elite 16–18, boss 19–21) |
| Episodes | hang table **`n_eps=20`** → 3×6×20 = **360** fights. Smoke `n_eps=1` is 18 fights, not the table. |
| Seed | `40000 + fixture_index*1000 + enc_id*100 + ep` (`env.reset(seed=…)`) |
| Win | `terminated and reward > 0` (sparse +1 win / −1 loss) |
| Max gym steps | 400 |
| Relics | Fixture `relics` applied into `CombatState`. HOLD 01–03 lock **`BURNING_BLOOD` + `SHURIKEN`**. If a fixture omits `relics`, HOLD injects those two (hang reconstruction). |
| Potions | Fixture `potions` applied when present (ids or `{id|potion_id|name}`). Not invented when omitted. |
| Reset path | Hang-era `options_from_fixture` → `STS2CombatEnv.reset(..., options={deck,hp,max_hp,relics,potions})`. `loadout_provider` also applies the same keys. |
| PYTHONPATH | Repo checkout `sts2_env` (`PYTHONPATH=.` or editable install). Do not mix a stray `/workspace/sts2-sim` copy of `eval_combat_suite.py` (bare 22-enc, no `--suite loadout_v1`). |

HOLD 01–03 JSON now carry `"relics": ["BURNING_BLOOD", "SHURIKEN"]`. Hang-era fixtures also had potions on some snapshots; this runner applies them when the JSON has a `potions` list.

## Runners (same protocol)

Full hang table (n=20):

```bash
PYTHONPATH=. python scripts/eval_combat_suite.py \
  --suite loadout_v1 --n-eps 20 \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --out evals/obs_v1_bh_v1_loadout_v1_n20.summary.json
```

Equivalent 0-step smoke/table via the library (same relics/options path):

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from sb3_contrib import MaskablePPO
from sts2_env.eval.combat_hold import run_hold_smoke, HUNG_COMBAT_ZIP
zip_path = Path(HUNG_COMBAT_ZIP)
model = MaskablePPO.load(str(zip_path), device='cpu')
def predict_fn(obs, mask):
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)
s = run_hold_smoke(predict_fn, n_eps=20)
print(s['overall']['win_rate'], s['elite']['win_rate'], s['boss']['win_rate'], s['passed'])
"
```

Expect overall ≈74% and Boss ≈49% on `bh_v1` (tolerance ~±3–5pp). Gate remains **≥70 / Boss≥40** on this protocol, not on the relic-stripped path.

## Not this protocol

- Bare Act1 22-enc (`eval_combat_suite.py` without `--suite loadout_v1`).
- Act1 RunEnv hierarchical (`scripts/eval_act1_runenv.py`).
- Continue-train / mix fracs. HOLD alignment is eval plumbing, not a new recipe.
