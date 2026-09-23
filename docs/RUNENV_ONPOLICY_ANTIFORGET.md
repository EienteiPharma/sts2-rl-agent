# RunEnv on-policy anti-forgetting mix

`combat_runenv_onpolicy_v1` (500k **pure** hang-protocol RunEnv combat from `bh_v1`) is **frozen**. Result: Act1 RunEnv **4%/med7** (over hang bar 3%/6.5) but loadout_v1 HOLD **68.9 / 98.3 / 39.4 FAIL** (need overall ≥70 and Boss ≥40). Classic fixture forgetting. Hang zip stays `bh_v1`. Do not rerun that pure recipe.

This knife **mixes** hang-protocol RunEnv combat segments with `loadout_v1` fixture combats in one MaskablePPO continue-from `bh_v1`, so RunEnv gains do not trash HOLD.

Eval hang flags/bars unchanged. EVENT / neow Jev stay off on the RunEnv half. MAP/CARD 0.65 and REST soft code untouched. **Not** a win claim. Do not swap the hang zip until Act1 clear **≥5%** **and** loadout_v1 HOLD ≥70 / Boss≥40 (评测哨兵).

## Mix

`MixedHangLoadoutEnv` (`sts2_env/gym_env/runenv_antiforget.py`):

- Same combat `OBS_SIZE=181` / `ACTION_SPACE_SIZE` as `bh_v1`.
- On each Gym `reset()`, pick hang `RunEnvOnPolicyCombatEnv` with probability `--runenv-frac` (default **0.70**); else `STS2CombatEnv` + rotating `loadout_v1` train fixtures (the 50 PR LOCKED decks, **not** `mix_neow_v1`).
- RunEnv half: auto-step noncombat with hang Jev (EVENT off, Neow random, `--start-with-neow`). PPO timestep = combat step only.
- Loadout half: ordinary fixture combat (all Act1 encounters). Replay for HOLD.

**Why 70/30 not 50/50:** the OOD problem is hang-protocol natural decks; 70% of episodes stay on that distribution. 30% `loadout_v1` is enough replay that 500k pure forgot (Boss 39.4). `n_envs=1` still mixes because sampling is per episode, not per VecEnv slot. 50/50 is `--runenv-frac 0.5`.

Continue-from **`bh_v1`**, not `onpolicy_v1` (that zip already failed HOLD). `onpolicy_v1` is allowed only as an explicit `--continue-from` if Surplus wants to try rescue; default does not.

## HOLD smoke callback

`sts2_env/eval/combat_hold.py` — same layout as box `eval_combat_suite --suite loadout_v1`:

- Fixtures `loadout_v1_01/02/03`
- Encounters 16–21 (elite+boss)
- Gate: overall **≥0.70**, Boss **≥0.40**

`--hold-freq N` runs that smoke every N combat timesteps, writes `hold_logs.json`, saves `best_hold/best_model` when a smoke **passes** and improves. `--hold-stop` ends `learn()` on a miss (off by default; n_eps=1 is noisy). `--hold-n-eps 1` is smoke (18 fights); `20` is full box HOLD (360).

## CLI

`scripts/train_combat_runenv_antiforget.py`

| Flag | Default | Notes |
|------|---------|--------|
| `--continue-from` / `--model` | hung `bh_v1` zip | obs_v1=181 |
| `--output-dir` | `output/combat_runenv_antiforget` | refuses `bh_v1` and `combat_runenv_onpolicy_v1` |
| `--runenv-frac` | `0.7` | P(RunEnv episode); rest `loadout_v1` |
| `--lr` | `3e-5` | Same fine-tune LR as frozen on-policy |
| `--total-timesteps` | `2048` | Smoke; first full recipe **250000** |
| `--n-envs` | `1` | `>1` uses `SubprocVecEnv` of pickle-friendly `MixedHangLoadoutEnvMaker`. Each worker still hang-protocol Jev on the RunEnv half (TypeSafe stays on the learn path — that is the ~8fps). Prefer `docs/RUNENV_COMBAT_OFFLINE.md` to collect combat segments first. |
| `--hold-freq` | `0` | `0` skips HOLD callback |
| `--hold-n-eps` | `1` | Smoke HOLD |
| `--hold-stop` | off | Optional fail-closed |
| `--dry-run` | off | Both mix halves + HOLD job count; no torch |

## Smoke (box CPU)

```bash
python scripts/train_combat_runenv_antiforget.py --dry-run

python scripts/train_combat_runenv_antiforget.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_antiforget_smoke \
  --runenv-frac 0.7 --n-envs 1 --n-steps 64 --batch-size 64 --total-timesteps 256
```

## Recommended first full recipe (Surplus)

Shorter than the frozen 500k pure, same LR, 70/30 mix, HOLD smoke every 25k:

```bash
python scripts/train_combat_runenv_antiforget.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_antiforget_v1 \
  --runenv-frac 0.7 --lr 3e-5 \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

If HOLD holds but Act1 clear is still &lt;5%, bump `--total-timesteps 500000` **with the mix** (do not go back to pure RunEnv). If HOLD drops, raise loadout replay (`--runenv-frac 0.5`) rather than another fixture-only `train_combat.py` run.

After train: hang-protocol Act1 eval on the new zip **and** loadout_v1 HOLD. 评测哨兵 owns both gates. Hang stays `bh_v1` until both pass.

## Parallel envs

`--n-envs > 1` is `SubprocVecEnv(makers)` of `MixedHangLoadoutEnvMaker` (not nested script closures). Linux spawn/fork can pickle that maker. **Each** process still auto-steps MAP/REST/CARD with hang Jev. Raising `n_envs` parallelizes collection; it does not turn Jev off.

To keep TypeSafe off `MaskablePPO.learn`, use the two-phase path in `docs/RUNENV_COMBAT_OFFLINE.md` (`collect_runenv_combat.py` then `train_combat_from_buffer.py`). Online mix above remains valid when Surplus wants live on-policy RunEnv.
