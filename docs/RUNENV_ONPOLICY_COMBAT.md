# RunEnv on-policy combat micro

**Frozen recipe:** `combat_runenv_onpolicy_v1` (500k **pure** hang-protocol RunEnv from `bh_v1`) hit Act1 **4%/med7** but loadout_v1 HOLD **FAIL**. Do not rerun that pure 500k. Hang zip stays `bh_v1`. Next knife: `docs/RUNENV_ONPOLICY_ANTIFORGET.md`.

Fine-tune the hung combat zip on **hang-protocol `STS2RunEnv` rollouts**, learning **combat-phase transitions only**.

This is **not** another loadout-fixture train (`train_combat.py` / `loadout_v*` / `mix_neow_v1`). Those already HOLD the combat suite and did **not** move Act1 RunEnv clear. This is **not** `train_full_run.py` (full RunEnv action space / RunEnv obs).

Eval hang protocol is **locked**. Do not change bars or flags here. Do **not** swap the hang zip in eval docs until Act1 RunEnv clear **≥5%** on that protocol (评测哨兵).

## Hang protocol (frozen)

See `docs/HANG_PROTOCOL_2026-09-22.md`.

- Combat init: `bh_v1` zip  
  `/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip`
- Hierarchical: MAP / REST / CARD Jev (`--jev on`; box `--jev-mode suggest_live`)
- `--jev-event off` `--jev-neow off` `--start-with-neow`
- Eval bar (unchanged): clear **3%** / median **6.5**. Win gate: clear **≥5%**.

## Approach

`RunEnvOnPolicyCombatEnv` (`sts2_env/gym_env/runenv_onpolicy_combat.py`) wraps `STS2RunEnv`:

1. Reset with `options={"start_with_neow": True}` (forced).
2. Auto-step every **non-combat** decision with `choose_jev_noncombat` and hang `JevPolicyFlags` (EVENT off, Neow random). Same helper as `scripts/eval_act1_runenv.py`. Fail-open: missing TypeSafe key → legal random, still a decision.
3. Expose **combat** `OBS_SIZE=181` and `ACTION_SPACE_SIZE` + `get_action_mask` to MaskablePPO (compatible with `bh_v1`).
4. Map PPO action `local` → RunEnv `_COMBAT_START + local`.
5. Reward = `compute_reward` on the combat that was stepped (`+1` win / `-1` loss / `0` else). RunEnv’s run-win/death reward is **not** the PPO target.
6. After a combat ends in a win, auto-step noncombat until the **next** combat in the same run (Neow + natural deck under hang protocol). Loss or run-over ends the Gym episode.

PPO never sees MAP / REST / CARD / EVENT / shop / Neow as learning steps. Those transitions have **zero gradient**.

Episode = one full run with multiple combats and frozen noncombat between them.

## What counts as a combat step

A PPO timestep is one `STS2RunEnv.step` while `mgr.phase == PHASE_COMBAT`. Inner RunEnv bookkeeping after a finishing blow is not a second PPO step. Auto-noncombat between fights is not a PPO step.

## CLI

`scripts/train_combat_runenv_onpolicy.py`

| Flag | Default | Notes |
|------|---------|--------|
| `--continue-from` / `--model` | hung `bh_v1` zip | Existing combat MaskablePPO (obs_v1=181) |
| `--output-dir` | `output/combat_runenv_onpolicy` | **Required new dir**; refuses `combat_ppo_obs_v1_bh_v1` |
| `--n-envs` | `1` | Box CPU |
| `--n-steps` | `128` | Combat steps per PPO rollout |
| `--total-timesteps` | `2048` | Smoke-scale; **not** a multi-hour default |
| `--batch-size` | `64` | Must be ≤ `n_steps * n_envs` |
| `--lr` | `3e-5` | Fine-tune (lower than from-scratch `3e-4`) |
| `--eval-freq` | `0` | `0` skips eval callback |
| `--dry-run` | off | Construct env + hang flags; no SB3 `learn` |
| `--loadout` | **absent** | Forbidden |

`MaskablePPO.load(..., env=train_env)` then `learn(..., reset_num_timesteps=False)`.

## Smoke (box CPU, minutes)

```bash
pip install -e ".[train]"
python scripts/train_combat_runenv_onpolicy.py --dry-run

python scripts/train_combat_runenv_onpolicy.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_onpolicy_smoke \
  --n-envs 1 --n-steps 64 --batch-size 64 --total-timesteps 256
```

`--dry-run` does not need torch. The short `learn` needs `sts2-rl-agent[train]`.

## Full train (Surplus box)

```bash
python scripts/train_combat_runenv_onpolicy.py \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_onpolicy_v1 \
  --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 4 \
  --lr 3e-5 --total-timesteps 500000 --eval-freq 10000 --eval-episodes 8
```

Bump `--n-envs` only if the box wants more CPU. Write a **new** outdir. After train, eval with the locked hang command on that zip; 评测哨兵 owns the ≥5% gate. Do not edit hang bars or swap `bh_v1` in eval docs from this trainer.

The 500k command above is the **frozen** pure recipe (`combat_runenv_onpolicy_v1`). Surplus should not launch it again; use the anti-forgetting mix instead.
