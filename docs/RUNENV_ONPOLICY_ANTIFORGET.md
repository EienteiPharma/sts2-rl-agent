# RunEnv on-policy anti-forgetting mix (loadout-dominant)

`combat_runenv_onpolicy_v1` (500k **pure** hang-protocol RunEnv combat from `bh_v1`) is **frozen**. Result: Act1 RunEnv **4%/med7** (over hang bar 3%/6.5) but loadout_v1 HOLD **68.9 / 98.3 / 39.4 FAIL** (need overall ≥70 and Boss ≥40). Hang zip stays `bh_v1`. Do not rerun that pure recipe.

`combat_runenv_antiforget_v1` (`--runenv-frac 0.7` online mix from `bh_v1`) is **frozen**. Result: HOLD **50.0 / 77.8 / 22.2 FAIL** (worse than pure on-policy). RunEnv 3%/med7 — no hang swap. **Do not continue-from the antiforget_v1 zip. Do not rerun 0.7.**

This knife **mixes** hang-protocol RunEnv combat with `loadout_v1` fixture combats in one MaskablePPO continue-from **`bh_v1` only**. Default **`--runenv-frac 0.3`**. Goal: HOLD recovers while still seeing some hang-protocol RunEnv combat.

**Episode vs step skew (loadoutdom_ep_v1):** `MixedHangLoadoutEnv.reset` used to sample the source with `P(runenv)=runenv_frac` **per episode**. Hang/buffer combats are ~50× longer than loadout fixture fights (~210 vs ~5 steps). Measured on real `runenv_combat_buffer_ep` (50k) + loadout_v1:

| runenv_frac | ep_runenv | **step_runenv** |
|---|---|---|
| 0.15 | ~0.16 | **~0.87** |
| 0.30 | ~0.29 | **~0.94** |
| 0.40 | ~0.41 | **~0.96** |
| 0.70 | ~0.72 | **~0.99** |

So "loadout-dominant 0.3" was ~94% buffer timesteps. HOLD collapsed (**loadoutdom_ep_v1** overall 39 / Boss 2.8). That recipe is **frozen** as an episode-mix result. Do **not** invent new fracs until `--mix-by steps` is on the tip and Surplus re-runs the same knifes.

Eval hang flags/bars unchanged. EVENT / neow Jev stay off on the RunEnv half. MAP/CARD 0.65 and REST soft code untouched. **Not** a win claim. Dual gate unchanged: do not swap the hang zip until Act1 clear **≥5%** **and** loadout_v1 HOLD ≥70 / Boss≥40 (评测哨兵).

## Mix

`MixedHangLoadoutEnv` (`sts2_env/gym_env/runenv_antiforget.py`):

- Same combat `OBS_SIZE=181` / `ACTION_SPACE_SIZE` as `bh_v1`.
- **`--mix-by steps`** (from_buffer / `CombatReplayEnv` default): on each `reset()`, pick the source that drives **cumulative PPO step fraction** toward `--runenv-frac` (under-represented source wins, ±0.05 jitter). This is the knife after loadoutdom_ep_v1.
- **`--mix-by episodes`**: Bernoulli per reset with `P(runenv)=runenv_frac`. Live online RunEnv half still defaults to this. Do not use for buffer training.
- Else `STS2CombatEnv` + rotating `loadout_v1` train fixtures (the 50 PR LOCKED decks, **not** `mix_neow_v1`).
- RunEnv half: auto-step noncombat with hang Jev (EVENT off, Neow random, `--start-with-neow`). PPO timestep = combat step only. Offline path uses collected combat segments instead (no Jev on `learn`).
- Loadout half: ordinary fixture combat (all Act1 encounters). Replay for HOLD.

`--runenv-frac 0.7` still parses but prints a freeze warning; do not launch it. Continue-from **`bh_v1` only**. Trainers refuse any `--continue-from` path containing `antiforget_v1`. They also refuse to write `bh_v1` / `combat_runenv_onpolicy_v1` / `combat_runenv_antiforget_v1` outdirs.

## HOLD smoke callback

`sts2_env/eval/combat_hold.py` — locked HOLD protocol (`docs/HOLD_PROTOCOL.md`):

- Fixtures `loadout_v1_01/02/03` with relics `BURNING_BLOOD` + `SHURIKEN` (applied via reset options; omitted relics default to those two)
- Encounters 16–21 (elite+boss)
- Seeds `40000+fix*1000+enc*100+ep`
- Gate: overall **≥0.70**, Boss **≥0.40** on this protocol (hang table 74.2 / 98.9 / Boss 49.4). Relic-stripped HOLD is not the gate.

`--hold-freq N` runs that smoke every N combat timesteps, writes `hold_logs.json`, saves `best_hold/best_model` when a smoke **passes** and improves. `--hold-stop` ends `learn()` on a miss (off by default; n_eps=1 is noisy). `--hold-n-eps 1` is smoke (18 fights); `20` is full box HOLD (360).

## CLI (online mix)

`scripts/train_combat_runenv_antiforget.py`

| Flag | Default | Notes |
|------|---------|--------|
| `--continue-from` / `--model` | hung `bh_v1` zip | obs_v1=181; refuses `antiforget_v1` |
| `--output-dir` | `output/combat_runenv_antiforget_ld03` | refuses `bh_v1`, `combat_runenv_onpolicy_v1`, `combat_runenv_antiforget_v1` |
| `--runenv-frac` | `0.3` | P(RunEnv episode); rest `loadout_v1`. **0.7 frozen** |
| `--lr` | `3e-5` | Keep this or lower |
| `--total-timesteps` | `2048` | Smoke; first full recipe **250000** |
| `--n-envs` | `1` | `>1` uses `SubprocVecEnv` of pickle-friendly `MixedHangLoadoutEnvMaker`. Each worker still hang-protocol Jev on the live RunEnv half. Prefer `docs/RUNENV_COMBAT_OFFLINE.md` to collect first. |
| `--hold-freq` | `0` | `0` skips HOLD callback |
| `--hold-n-eps` | `1` | Smoke HOLD |
| `--hold-stop` | off | Optional fail-closed |
| `--dry-run` | off | Both mix halves + HOLD job count; no torch |

## Smoke (box CPU)

```bash
python scripts/train_combat_runenv_antiforget.py --dry-run

python scripts/train_combat_runenv_antiforget.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_antiforget_ld03_smoke \
  --runenv-frac 0.3 --n-envs 1 --n-steps 64 --batch-size 64 --total-timesteps 256
```

## Recommended Surplus path (EP CUDA, `--mix-by steps`)

Prefer **existing combat buffer** + `train_combat_from_buffer.py` so TypeSafe is off `learn()`. Continue-from `bh_v1`. `--device auto`. **Re-run loadoutdom 0.3 with `--mix-by steps` into a new outdir** — do not invent new fracs. Episode-mix loadoutdom_ep_v1 is frozen.

```bash
python scripts/train_combat_from_buffer.py --dry-run --device auto --mix-by steps

python scripts/train_combat_from_buffer.py \
  --buffer output/runenv_combat_buffer/transitions.npz \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_offline_ld03_steps \
  --runenv-frac 0.3 --mix-by steps --lr 3e-5 --device auto \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

Same-frac re-runs (optional, still `--mix-by steps`, new outdirs): `--runenv-frac 0.4`, `0.15`, `0.0` (AL pure loadout; mix-by is a no-op). Do not continue-from loadoutdom_ep_v1.

## Two-phase

No extra helper. Continue-from **`bh_v1`**, never from a failed mix zip. Use **`--mix-by steps`** on the buffer path. Do not invent a new frac until the 0.3 step-mix re-run lands. Do not go back to `--runenv-frac 0.7`.

Online mix at 0.3 remains valid when Surplus wants live on-policy RunEnv (Jev on the learn path, ~8fps). Live mix still defaults **`--mix-by episodes`**.

```bash
python scripts/train_combat_runenv_antiforget.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_antiforget_ld03 \
  --runenv-frac 0.3 --lr 3e-5 \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

Do not go back to `--runenv-frac 0.7`. Do not go back to pure RunEnv. Do not run another fixture-only `train_combat.py` unless 评测哨兵 asks.

After train: hang-protocol Act1 eval on the **new** zip **and** loadout_v1 HOLD. 评测哨兵 owns both gates. Hang stays `bh_v1` until both pass.

## Parallel envs

`--n-envs > 1` is `SubprocVecEnv(makers)` of `MixedHangLoadoutEnvMaker` (not nested script closures). Linux spawn/fork can pickle that maker. **Each** live-RunEnv process still auto-steps MAP/REST/CARD with hang Jev. Raising `n_envs` parallelizes collection; it does not turn Jev off. Optional TypeSafe key pool (see `docs/RUNENV_COMBAT_OFFLINE.md`) assigns `keys[worker % n]`.

To keep TypeSafe off `MaskablePPO.learn`, use the two-phase collect then train-from-buffer path. First collect recipe: `--n-envs 2–4`, not 16.
