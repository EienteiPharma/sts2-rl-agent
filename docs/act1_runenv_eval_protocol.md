# Act1 RunEnv 评测口径（冻结）

**状态：LOCKED 2026-09-22** · 评分员：评测哨兵  
**范围：** Act1 整局通关评测（`STS2RunEnv`），**≠** combat 套件（bare / loadout_v0 / loadout_v1）。

## 种子集

| 项 | 值 |
|----|----|
| 种子 | `200000 .. 200049`（含端点，共 **50** 局） |
| 注入 | `env.reset(seed=S)`（Gymnasium 种子；同种确定性已冒烟） |
| 角色 | Ironclad |
| 进阶 | 0 |
| `max_steps` | 2000（env 默认） |

## 主指标 / 辅指标

1. **主：`act1_clear_rate`** — 当局过程中 `max(info['act']) >= 1` 的比例（打过 Act1 Boss 并进入 Act2 地图）。
2. **辅：**
   - `median_floor` / `mean_floor`（`info['floor']` 终局）
   - `mean_hp_on_clear`（仅通关局终局 `hp`；未通关不计入）
   - `full_run_win_rate`（`terminated and reward > 0`，三幕全通；旁证）
   - `trunc_rate`（`truncated`）
   - `error_rate`（step 异常被 env 吞成死亡的局，若 info 可区分则记；否则并入失败）

## 禁止混谈

- **不得**把 bare / loadout_* combat 胜率写成 Act1 通关率。
- loadout_v1 战斗套件报赢（74.2/98.9/49.4）**不**自动等于本表通关。
- Jev 战略 Choice 由 `--jev on` 接入非战斗决策；阈值见下，**不**等于 Act1 通关完成。
- **`hierarchical` ≠ Act1 通关完成。** 战斗步只接 hung combat zip；非战斗默认合法随机。通关门禁仍由实验室另挂。

## 策略接口

脚本：`scripts/eval_act1_runenv.py`

```bash
# random 基线
python scripts/eval_act1_runenv.py --policy random --out /workspace/sts2-sim/evals/act1_runenv_random_s200000.json

# RunEnv 兼容 MaskablePPO（obs=RUN_OBS_SIZE，非 combat zip）
python scripts/eval_act1_runenv.py --policy model --model path/to/run_model.zip --out ...

# hierarchical：战斗步用 combat zip（obs=OBS_SIZE / obs_v1=181）；非战斗默认合法随机
python scripts/eval_act1_runenv.py --policy hierarchical --combat-model path/to/combat_ppo_obs_v1.zip --jev off

# hierarchical + Jev（非战斗 Choice / rest_or_continue Score）；combat zip 不变
python scripts/eval_act1_runenv.py --policy hierarchical --combat-model path/to/combat_ppo_obs_v1.zip --jev on
```

- `--policy random|model|hierarchical`
- `--model`：仅 `--policy model`。必须 `obs_dim == RUN_OBS_SIZE`（当前 201）。combat-only zip **拒评**并明文报错，不静默。
- `--combat-model`：仅 `--policy hierarchical`。必须 `obs_dim == OBS_SIZE`（obs_v1 = 181）。**禁止**把 RunEnv obs 喂给 combat zip。
- `--jev off|on`（默认 `off`；`--strategic jev` 等同 `--jev on`）。仅 hierarchical。
- 战斗步：从 `RunManager.get_combat_state()` 取 `CombatState`，`encode_observation(combat)` + combat `get_action_mask`，`MaskablePPO.predict`，再把 combat action index 映射到 RunEnv combat slice（layout offset 0）。**Jev 不进战斗。**
- 非战斗步：
  - `--jev off`：`action_masks()==1` 合法随机；日志 `shadow_status=stub`。
  - `--jev on`：TypeSafe/Jev Choice（`map_fork` / `card_reward` / 同类决策）与 rest_or_continue 的 hp_pressure Score。Choice confidence **≥ 0.65** 否则 uncertain → 合法随机。hp_pressure **≥ 2.0** 优先 rest，**≤ 1.0** 优先 continue，中间信 Choice。先剥 invisible/illegal。Act1 reward `+` 卡不是自然掉落（仅 Smith/Neow）——criteria 见 `docs/act1_content_map.md`。真 `pick_card` 另打 4 档 Score `card_fit`：**不改** Choice 0.65；confidence < 0.65 但 `card_fit >= 2.0` 且 choice ≠ skip 时落地，原因 `jev_card_fit_assist`。`PHASE_CARD_REWARD` 上的 `pick_potion` / `pick_relic_reward` **不**走 card_reward Jev，合法随机原因 `potion_or_relic_reward_random`（不进 CARD 落地率分母）。API/模型错误记 `shadow_status=error` 并回退合法随机，**不吞决策点**。
  - `--jev on` 时日志含 `shadow_suggestion`、`shadow_confidence`、`shadow_fallback_reason`（及 `shadow_hp_pressure`）；真卡屏可选 `jev_card_fit`。
- 输出：完整 JSON + 同名 `.summary.json`（无逐局 rows 的精简版）。
- Live 调用读 `TYPESAFE_API_KEY`（`sts2_env/eval/jev.py` LiveJevClient）。密钥缺失时按 error 回退，不改 combat 路径。TODO(Surplus/Jev) 可替换该 adapter，不必改 eval 脚本。

## 观测尺寸（obs_v1）

| 向量 | 宽度 | 用途 |
|------|------|------|
| Combat `OBS_SIZE` | **181** | 全量 `IntentType` one-hot；hung combat zip（`combat_ppo_obs_v1_*`） |
| RunEnv `RUN_OBS_SIZE` | **201** | 181 combat + 20 run-state extras |

两套观测不可互换。`--model` 与 `--combat-model` 分轨就是为了挡住误喂。

## 门禁

本表门禁由实验室另挂；冻结时门禁为空。未出通关数字前不谈「落地完成」。
hierarchical 评测通过 **不等于** Act1 通关达标。
