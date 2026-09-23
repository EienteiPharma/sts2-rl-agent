# RunEnv on-policy anti-forgetting mix (loadout-dominant)

`combat_runenv_onpolicy_v1` (500k **pure** hang-protocol RunEnv combat from `bh_v1`) is **frozen**. Result: Act1 RunEnv **4%/med7** (over hang bar 3%/6.5) but loadout_v1 HOLD **68.9 / 98.3 / 39.4 FAIL** (need overall ≥70 and Boss ≥40). Hang zip stays `bh_v1`. Do not rerun that pure recipe.

`combat_runenv_antiforget_v1` (`--runenv-frac 0.7` online mix from `bh_v1`) is **frozen**. Result: HOLD **50.0 / 77.8 / 22.2 FAIL** (worse than pure on-policy). RunEnv 3%/med7 — no hang swap. **Do not continue-from the antiforget_v1 zip. Do not rerun 0.7.**

This knife **mixes** hang-protocol RunEnv combat with `loadout_v1` fixture combats in one MaskablePPO continue-from **`bh_v1` only**, default **`--runenv-frac 0.3`** (≥70% loadout). Goal: HOLD recovers while still seeing some hang-protocol RunEnv combat.

Eval hang flags/bars unchanged. EVENT / neow Jev stay off on the RunEnv half. MAP/CARD 0.65 and REST soft code untouched. **Not** a win claim. Dual gate unchanged: do not swap the hang zip until Act1 clear **≥5%** **and** loadout_v1 HOLD ≥70 / Boss≥40 (评测哨兵).

## Mix

`MixedHangLoadoutEnv` (`sts2_env/gym_env/runenv_antiforget.py`):

- Same combat `OBS_SIZE=181` / `ACTION_SPACE_SIZE` as `bh_v1`.
- On each Gym `reset()`, pick hang `RunEnvOnPolicyCombatEnv` (or buffer replay) with probability `--runenv-frac` (default **0.30**); else `STS2CombatEnv` + rotating `loadout_v1` train fixtures (the 50 PR LOCKED decks, **not** `mix_neow_v1`).
- RunEnv half: auto-step noncombat with hang Jev (EVENT off, Neow random, `--start-with-neow`). PPO timestep = combat step only. Offline path uses collected combat segments instead (no Jev on `learn`).
- Loadout half: ordinary fixture combat (all Act1 encounters). Replay for HOLD.

**Why 0.3 not 0.7:** 70% RunEnv trashed HOLD (50.0/77.8/22.2). Loadout-dominant (30% hang combat / 70% loadout_v1) is the knife. `--runenv-frac 0.7` still parses but prints a freeze warning; do not launch it. If HOLD still misses after 0.3, drop to `--runenv-frac 0.1` from `bh_v1` (do not continue-from a failed mix zip). `n_envs=1` still mixes because sampling is per episode.

Continue-from **`bh_v1` only**. Trainers refuse any `--continue-from` path containing `antiforget_v1`. They also refuse to write `bh_v1` / `combat_runenv_onpolicy_v1` / `combat_runenv_antiforget_v1` outdirs.

## HOLD smoke callback

`sts2_env/eval/combat_hold.py` — same layout as box `eval_combat_suite --suite loadout_v1`:

- Fixtures `loadout_v1_01/02/03`
- Encounters 16–21 (elite+boss)
- Gate: overall **≥0.70**, Boss **≥0.40**

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

## Recommended Surplus path (EP CUDA, smallest)

Prefer **existing combat buffer** + `train_combat_from_buffer.py` so TypeSafe is off `learn()`. Continue-from `bh_v1`. `--device auto` (cuda if available).

```bash
python scripts/train_combat_from_buffer.py --dry-run --device auto

python scripts/train_combat_from_buffer.py \
  --buffer output/runenv_combat_buffer/transitions.npz \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_offline_ld03 \
  --runenv-frac 0.3 --lr 3e-5 --device auto \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

If the buffer is missing, collect once (hang Jev still on collect; `--n-envs 2–4`):

```bash
python scripts/collect_runenv_combat.py \
  --out output/runenv_combat_buffer/transitions.npz \
  --n-envs 4 --n-steps 50000 --policy model \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip
```

Online mix at 0.3 remains valid when Surplus wants live on-policy RunEnv (Jev on the learn path, ~8fps):

```bash
python scripts/train_combat_runenv_antiforget.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_antiforget_ld03 \
  --runenv-frac 0.3 --lr 3e-5 \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

## Two-phase (if 0.3 HOLD still misses)

No extra helper. Both phases continue-from **`bh_v1`**, never from a failed mix zip.

1. Short RunEnv exposure (loadout still majority), then HOLD smoke.
2. If overall &lt;70 or Boss &lt;40, restart from `bh_v1` with `--runenv-frac 0.1` (90% loadout).

```bash
# Phase 1 — already the default full recipe above (frac 0.3, 250k).

# Phase 2 — HOLD recovery from bh_v1 (do not continue-from ld03 if HOLD failed)
python scripts/train_combat_from_buffer.py \
  --buffer output/runenv_combat_buffer/transitions.npz \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_offline_ld01 \
  --runenv-frac 0.1 --lr 3e-5 --device auto \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

Do not go back to `--runenv-frac 0.7`. Do not go back to pure RunEnv. Do not run another fixture-only `train_combat.py` unless 评测哨兵 asks.

After train: hang-protocol Act1 eval on the **new** zip **and** loadout_v1 HOLD. 评测哨兵 owns both gates. Hang stays `bh_v1` until both pass.

## Parallel envs

`--n-envs > 1` is `SubprocVecEnv(makers)` of `MixedHangLoadoutEnvMaker` (not nested script closures). Linux spawn/fork can pickle that maker. **Each** live-RunEnv process still auto-steps MAP/REST/CARD with hang Jev. Raising `n_envs` parallelizes collection; it does not turn Jev off. Optional TypeSafe key pool (see `docs/RUNENV_COMBAT_OFFLINE.md`) assigns `keys[worker % n]`.

To keep TypeSafe off `MaskablePPO.learn`, use the two-phase collect then train-from-buffer path. First collect recipe: `--n-envs 2–4`, not 16.
