# Colab notebooks (GPU parallel track)

**Drive convention (Pharma default):** mount `drive` → read/write under **`/content/drive/MyDrive/sts2/colab/`** (features parquet/npz, critic `.pt`); set notebook `USE_DRIVE=False` for `/content/sts2/colab/` fallback when Drive is unavailable.

## `colab_combat_feature_extract.ipynb`

**Scope (tip #1):** feature skeleton only — read a local `transitions.npz`, validate,
subsample ≥1000 SARS rows, write parquet/npz. **No** game comms, EP bridge,
`sts2.dll`, or live client.

**Input npz keys:** `obs`, `next_obs`, `action`, `reward`, `done`, `action_mask`.
`obs` / `next_obs` are `(N, 181)` float32 (combat obs_v1).

**Gates:** ≥1000 exported rows; assert no NaN/Inf in numeric arrays.

**Optional context (not a gate for #1):** HOLD turn replay JSONL
(`hold_turn_replay_*.jsonl`) may document hp/enemies_hp for later tips.

Canonical logic: `sts2_env/colab/combat_transition_features.py`.

## `colab_combat_critic.ipynb` (tip #2)

Minimal **value network / Critic** smoke on tip #1 features:

| | |
|--|--|
| **In** | `obs` `(181,)` float32 |
| **Out** | scalar value `(1,)` — MSE vs **MC return** (`G_t = r_t + 0.99·G_{t+1}`, reset at `done`) |
| **Ckpt** | `.pt` via `train_critic_smoke` → `sts2_env/colab/combat_critic.py` |
| **Gate** | ≥1000 rows, finite loss, no NaN |

CLI: `scripts/colab_train_critic_smoke.py FEATURES OUT.pt`.

## `colab_combat_critic_onnx.ipynb` (tip #3)

Export tip #2 `.pt` → **`combat_critic_smoke.onnx`** on Drive (`MyDrive/sts2/colab/`); ORT verify gate max abs err ≤ **1e-4**. CLI: `scripts/colab_export_critic_onnx.py CKPT OUT.onnx`.

## `colab_combat_critic_colab_v1.ipynb` (tip #4)

Formal critic on **full** colab_v1 `transitions.npz` (~500k); warm-start `combat_critic_smoke.pt` → `combat_critic_colab_v1.pt` + `.onnx` (ORT ≤ **1e-5**). CLI: `scripts/colab_train_critic_colab_v1.py`.
