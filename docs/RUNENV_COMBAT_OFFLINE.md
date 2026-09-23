# Offline hang-protocol combat collect → train-from-buffer

Online anti-forgetting mix (`docs/RUNENV_ONPOLICY_ANTIFORGET.md`) is **still valid**. It sits ~8fps because TypeSafe Jev HTTP is inside `RunEnvOnPolicyCombatEnv.step` / `reset` (auto-noncombat on the PPO learn path). This knife splits that:

1. **Collect** hang-protocol RunEnv rollouts; write **combat segments only**.
2. **Train** MaskablePPO from that buffer so Jev does **not** block `learn()`.

Hang protocol is **locked**. Do not turn Jev off, do not replace MAP/CARD with random, do not change hang flags/bars to buy fps. Dual gate unchanged: Act1 RunEnv clear **≥5%** **and** loadout_v1 HOLD overall ≥70 / Boss ≥40. Hang zip stays `bh_v1` until both pass. Not a win claim.

## Hang protocol (frozen)

See `docs/HANG_PROTOCOL_2026-09-22.md`.

- Continue-from: `bh_v1`  
  `/workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip`
- MAP / REST / CARD Jev (`--jev on`; box `suggest_live`)
- `--jev-event off` `--jev-neow off` `--start-with-neow`
- Collector uses `RunEnvOnPolicyCombatEnv` / `choose_jev_noncombat` (same as online). Each `--n-envs` worker is still hang-protocol Jev.
- First collect recipe: **`--n-envs 2–4`**, not 16.

## TypeSafe key pool (box-secrets; no re-paste)

Surplus stores keys at `/home/box/agent-data/box-secrets.json`:

- `card.TYPESAFE_API_KEY`
- `card.TYPESAFE_API_KEY_1` … `card.TYPESAFE_API_KEY_4`

Process env is **not** assumed to be pre-injected. `collect_runenv_combat.py` / `LiveJevClient` / `ensure_typesafe_api_key` call `load_typesafe_api_keys()`, which reads those `card.*` fields, injects missing names into the process env for workers, then also accepts already-exported `TYPESAFE_API_KEY` / `TYPESAFE_API_KEY_N` / `TYPESAFE_API_KEYS` (exported names are not overwritten). Logs **count only**. Never print or paste key material into chat.

On Cloudflare **1010** / HTTP **403** the client rotates to the next key and backs off ~1s. MAP/CARD stay Jev (not random). Round-robin: `keys[worker_id % len(keys)]`. First collect recipe: **`--n-envs 2–4`**.

```bash
# Surplus launch — no user re-paste. Pool loads from box-secrets automatically.
python scripts/collect_runenv_combat.py --dry-run

python scripts/collect_runenv_combat.py \
  --out output/runenv_combat_buffer/transitions.npz \
  --n-envs 4 --n-steps 50000 --policy model \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip
```

`--n-envs 4` with fewer keys round-robins. `--n-envs > 4` prints a warning.

Do **not** stop a running `combat_runenv_antiforget_v1` job to apply the pool. Online mix still defaults `--n-envs 1`.

## Buffer

`sts2_env/gym_env/combat_buffer.py` writes `transitions.npz` + `transitions.meta.json`.

| key | shape |
|-----|--------|
| `obs` | `(N, 181)` float32 |
| `next_obs` | `(N, 181)` float32 |
| `action` | `(N,)` int64 |
| `reward` | `(N,)` float32 |
| `done` | `(N,)` bool |
| `action_mask` | `(N, ACTION_SPACE_SIZE)` int8 |

`CombatReplayEnv` replays those rows. PPO `step(action)` advances the stored sequence (collector policy generated `next_obs` / reward). Re-collect when the policy you want on-policy data from has moved.

## Collect CLI

`scripts/collect_runenv_combat.py`

```bash
python scripts/collect_runenv_combat.py --dry-run

python scripts/collect_runenv_combat.py \
  --out output/runenv_combat_buffer/transitions.npz \
  --n-envs 1 --n-steps 32 --policy random

python scripts/collect_runenv_combat.py \
  --out output/runenv_combat_buffer/transitions.npz \
  --n-envs 4 --n-steps 50000 --policy model \
  --model /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip
```

| Flag | Default | Notes |
|------|---------|--------|
| `--out` | `output/runenv_combat_buffer/transitions.npz` | refuses `bh_v1` / `combat_runenv_onpolicy_v1` |
| `--n-steps` | `256` | Combat transitions (not noncombat auto-steps) |
| `--n-envs` | `1` | Parallel collectors; **each** still hang Jev. First recipe **2–4**, not 16 |
| `--policy` | `random` | `random` = legal mask (no torch). Surplus full collect: `model` |
| `--model` | hung `bh_v1` zip | Used only with `--policy model` |
| `--dry-run` | off | Hang env + flags + key **count**; no npz |

`--policy random` is smoke / CA. Surplus collect that should match `bh_v1` on-policy combat: `--policy model`.

## Train-from-buffer CLI

`scripts/train_combat_from_buffer.py`

Continue-from `bh_v1`. New outdir. Never overwrite `bh_v1` or frozen `combat_runenv_onpolicy_v1`. Optional `--runenv-frac` still mixes **live** `loadout_v1` (no Jev) with buffer replay (hang combat distribution).

```bash
python scripts/train_combat_from_buffer.py --dry-run

python scripts/train_combat_from_buffer.py \
  --buffer output/runenv_combat_buffer/transitions.npz \
  --continue-from /workspace/sts2-sim/output/combat_ppo_obs_v1_bh_v1/final_model.zip \
  --output-dir output/combat_runenv_offline_v1 \
  --runenv-frac 0.7 --n-envs 1 --n-steps 256 --batch-size 256 \
  --total-timesteps 250000 \
  --hold-freq 25000 --hold-n-eps 1
```

`--dry-run` does not need a buffer file (uses a synthetic 16-step tensor). Real `learn` needs `--buffer`. Default `--total-timesteps 2048` is smoke; do not launch 500k from CA.

`--n-envs > 1` uses `SubprocVecEnv` of pickle-friendly `MixedHangLoadoutEnvMaker(buffer_path=...)`. Replay half has **no** TypeSafe. Loadout half is fixture combat.

## Online mix vs this path

| Path | Jev | When |
|------|-----|------|
| `train_combat_runenv_antiforget.py` | On every RunEnv episode (`--n-envs` SubprocVecEnv, each worker hang Jev) | Live on-policy mix; ~8fps if TypeSafe is in `step` |
| collect + `train_combat_from_buffer.py` | Collect only | Prefer for Surplus fps; re-collect as the zip moves |

`--n-envs` on the **online** trainer already works (`SubprocVecEnv` + `MixedHangLoadoutEnvMaker`). That parallelizes Jev; it does not remove it. Do not set `--jev off` on hang eval to chase fps. Do **not** interrupt a running `antiforget_v1` (`--n-envs 1`) to apply a key pool; collect is the parallel path.

After train: hang-protocol Act1 eval on the **new** zip **and** loadout_v1 HOLD. 评测哨兵 owns both gates.
