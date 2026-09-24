# Colab notebooks (GPU parallel track)

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
